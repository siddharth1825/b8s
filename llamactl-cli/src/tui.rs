use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use llamactl_common::*;
use ratatui::{
    prelude::*,
    widgets::*,
};
use std::io::stdout;
use std::time::{Duration, Instant};

pub async fn run(server: &str) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let result = event_loop(&mut terminal, server).await;

    // Restore terminal
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;

    result
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    server: &str,
) -> Result<()> {
    let mut nodes: Vec<NodeInfo> = vec![];
    let mut deployments: Vec<DeploymentInfo> = vec![];
    let mut last_fetch = Instant::now() - Duration::from_secs(10); // force immediate fetch
    let mut error_msg: Option<String> = None;

    loop {
        // Poll for data every 2 seconds
        if last_fetch.elapsed() >= Duration::from_secs(2) {
            match fetch_data(server).await {
                Ok((n, d)) => {
                    nodes = n;
                    deployments = d;
                    error_msg = None;
                }
                Err(e) => {
                    error_msg = Some(e.to_string());
                }
            }
            last_fetch = Instant::now();
        }

        // Draw
        terminal.draw(|frame| {
            draw(frame, &nodes, &deployments, error_msg.as_deref());
        })?;

        // Handle input (non-blocking, 200ms poll)
        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                            return Ok(());
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

fn draw(
    frame: &mut Frame,
    nodes: &[NodeInfo],
    deployments: &[DeploymentInfo],
    error: Option<&str>,
) {
    let area = frame.area();

    // Layout: title, nodes table, deployments table, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // title
            Constraint::Min(5),    // nodes
            Constraint::Min(5),    // deployments
            Constraint::Length(1), // footer
        ])
        .split(area);

    // Title
    let title = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" llamactl top ")
        .title_alignment(Alignment::Center);

    let title_text = if let Some(err) = error {
        Line::from(vec![
            Span::styled("  ERROR: ", Style::default().fg(Color::Red).bold()),
            Span::raw(err),
        ])
    } else {
        let now = chrono::Local::now().format("%H:%M:%S");
        Line::from(vec![
            Span::styled(
                format!("  {} node(s)  ", nodes.len()),
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                format!("{}  deployment(s)  ", deployments.len()),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(
                format!("  refreshed {}", now),
                Style::default().fg(Color::DarkGray),
            ),
        ])
    };
    let title_widget = Paragraph::new(title_text).block(title);
    frame.render_widget(title_widget, chunks[0]);

    // Nodes table
    let node_header = Row::new(vec![
        "HOSTNAME", "STATUS", "BACKEND", "VRAM", "RAM", "CORES", "CPU%", "CACHED",
    ])
    .style(Style::default().fg(Color::Cyan).bold())
    .bottom_margin(0);

    let node_rows: Vec<Row> = nodes
        .iter()
        .map(|n| {
            let status_style = match n.status {
                NodeStatus::Online => Style::default().fg(Color::Green),
                NodeStatus::Offline => Style::default().fg(Color::Red),
                NodeStatus::Draining => Style::default().fg(Color::Yellow),
            };

            let vram = match (n.vram_used_mb, n.vram_mb) {
                (Some(u), Some(t)) => format!("{}/{} MB", u, t),
                (None, Some(t)) => format!("-/{} MB", t),
                _ => "-".to_string(),
            };

            let cpu_bar = make_bar(n.cpu_usage_pct as f64, 100.0);

            let cached = if n.cached_models.is_empty() {
                "-".to_string()
            } else {
                format!("{} model(s)", n.cached_models.len())
            };

            Row::new(vec![
                Cell::from(n.hostname.clone()),
                Cell::from(n.status.to_string()).style(status_style),
                Cell::from(n.backend.to_string()),
                Cell::from(vram),
                Cell::from(format!("{}/{} MB", n.ram_used_mb, n.total_ram_mb)),
                Cell::from(n.cpu_cores.to_string()),
                Cell::from(format!("{} {:.0}%", cpu_bar, n.cpu_usage_pct)),
                Cell::from(cached),
            ])
        })
        .collect();

    let nodes_table = Table::new(
        node_rows,
        [
            Constraint::Length(16),
            Constraint::Length(10),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Length(16),
            Constraint::Length(6),
            Constraint::Length(16),
            Constraint::Min(12),
        ],
    )
    .header(node_header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" Nodes ")
            .title_style(Style::default().fg(Color::White).bold()),
    );
    frame.render_widget(nodes_table, chunks[1]);

    // Deployments table
    let deploy_header = Row::new(vec![
        "MODEL", "NODE", "STATUS", "PORT", "ENDPOINT",
    ])
    .style(Style::default().fg(Color::Cyan).bold())
    .bottom_margin(0);

    let deploy_rows: Vec<Row> = deployments
        .iter()
        .map(|d| {
            let status_style = match d.status {
                DeploymentStatus::Running => Style::default().fg(Color::Green),
                DeploymentStatus::Pending | DeploymentStatus::Downloading => {
                    Style::default().fg(Color::Yellow)
                }
                DeploymentStatus::Failed => Style::default().fg(Color::Red),
                DeploymentStatus::Stopped => Style::default().fg(Color::DarkGray),
            };

            let model_short = if d.model.len() > 35 {
                format!("..{}", &d.model[d.model.len() - 33..])
            } else {
                d.model.clone()
            };

            Row::new(vec![
                Cell::from(model_short),
                Cell::from(d.node_hostname.clone()),
                Cell::from(d.status.to_string()).style(status_style),
                Cell::from(d.port.to_string()),
                Cell::from(d.endpoint.as_deref().unwrap_or("-").to_string()),
            ])
        })
        .collect();

    let deploy_table = Table::new(
        deploy_rows,
        [
            Constraint::Length(36),
            Constraint::Length(16),
            Constraint::Length(12),
            Constraint::Length(6),
            Constraint::Min(20),
        ],
    )
    .header(deploy_header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" Deployments ")
            .title_style(Style::default().fg(Color::White).bold()),
    );
    frame.render_widget(deploy_table, chunks[2]);

    // Footer
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" q", Style::default().fg(Color::Yellow).bold()),
        Span::raw(" quit  "),
    ]))
    .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(footer, chunks[3]);
}

/// Simple block-character bar: ████░░░░
fn make_bar(value: f64, max: f64) -> String {
    let width = 8;
    let filled = ((value / max) * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

async fn fetch_data(server: &str) -> Result<(Vec<NodeInfo>, Vec<DeploymentInfo>)> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()?;

    let (nodes_resp, deploy_resp) = tokio::join!(
        client.get(format!("{server}/api/nodes")).send(),
        client.get(format!("{server}/api/deployments")).send(),
    );

    let nodes: Vec<NodeInfo> = nodes_resp?.error_for_status()?.json().await?;
    let deployments: Vec<DeploymentInfo> = deploy_resp?.error_for_status()?.json().await?;

    Ok((nodes, deployments))
}
