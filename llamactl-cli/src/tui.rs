use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use llamactl_common::*;
use ratatui::{prelude::*, widgets::*};
use std::io::stdout;
use std::time::{Duration, Instant};

// ── App state ───────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Dashboard,
    NodeDetail,
    DeploymentDetail,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Panel {
    Nodes,
    Deployments,
}

struct App {
    server: String,
    view: View,
    panel: Panel,
    node_index: usize,
    deploy_index: usize,
    nodes: Vec<NodeInfo>,
    deployments: Vec<DeploymentInfo>,
    last_fetch: Instant,
    status_msg: Option<(String, Instant)>,
    error_msg: Option<String>,
    show_help: bool,
}

impl App {
    fn new(server: &str) -> Self {
        Self {
            server: server.to_string(),
            view: View::Dashboard,
            panel: Panel::Nodes,
            node_index: 0,
            deploy_index: 0,
            nodes: vec![],
            deployments: vec![],
            last_fetch: Instant::now() - Duration::from_secs(10),
            status_msg: None,
            error_msg: None,
            show_help: false,
        }
    }

    fn selected_node(&self) -> Option<&NodeInfo> {
        self.nodes.get(self.node_index)
    }

    fn selected_deployment(&self) -> Option<&DeploymentInfo> {
        self.deployments.get(self.deploy_index)
    }

    fn set_status(&mut self, msg: impl Into<String>) {
        self.status_msg = Some((msg.into(), Instant::now()));
    }
}

// ── Entry point ─────────────────────────────────────────────────

pub async fn run(server: &str) -> Result<()> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let result = event_loop(&mut terminal, server).await;

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    result
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    server: &str,
) -> Result<()> {
    let mut app = App::new(server);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()?;

    loop {
        // Fetch data every 2 seconds
        if app.last_fetch.elapsed() >= Duration::from_secs(2) {
            match fetch_data(&client, &app.server).await {
                Ok((n, d)) => {
                    app.nodes = n;
                    app.deployments = d;
                    app.error_msg = None;
                }
                Err(e) => app.error_msg = Some(e.to_string()),
            }
            app.last_fetch = Instant::now();
        }

        // Clear stale status messages after 3s
        if let Some((_, when)) = &app.status_msg {
            if when.elapsed() > Duration::from_secs(3) {
                app.status_msg = None;
            }
        }

        // Draw
        terminal.draw(|frame| draw(frame, &app))?;

        // Handle input
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if handle_input(&mut app, &client, key.code, key.modifiers).await {
                        return Ok(());
                    }
                }
            }
        }
    }
}

// ── Input handling ──────────────────────────────────────────────

async fn handle_input(
    app: &mut App,
    client: &reqwest::Client,
    code: KeyCode,
    modifiers: KeyModifiers,
) -> bool {
    // Global keys
    match code {
        KeyCode::Char('q') if app.view == View::Dashboard => return true,
        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => return true,
        KeyCode::Char('?') => {
            app.show_help = !app.show_help;
            return false;
        }
        KeyCode::Esc => {
            if app.show_help {
                app.show_help = false;
            } else {
                match app.view {
                    View::Dashboard => return true,
                    _ => app.view = View::Dashboard,
                }
            }
            return false;
        }
        _ => {}
    }

    if app.show_help {
        return false;
    }

    match app.view {
        View::Dashboard => handle_dashboard_input(app, client, code).await,
        View::NodeDetail => handle_node_detail_input(app, client, code).await,
        View::DeploymentDetail => handle_deployment_detail_input(app, client, code).await,
    }

    false
}

async fn handle_dashboard_input(app: &mut App, client: &reqwest::Client, code: KeyCode) {
    match code {
        KeyCode::Tab => {
            app.panel = match app.panel {
                Panel::Nodes => Panel::Deployments,
                Panel::Deployments => Panel::Nodes,
            };
        }
        KeyCode::Up | KeyCode::Char('k') => match app.panel {
            Panel::Nodes => app.node_index = app.node_index.saturating_sub(1),
            Panel::Deployments => app.deploy_index = app.deploy_index.saturating_sub(1),
        },
        KeyCode::Down | KeyCode::Char('j') => match app.panel {
            Panel::Nodes => {
                if !app.nodes.is_empty() {
                    app.node_index = (app.node_index + 1).min(app.nodes.len() - 1);
                }
            }
            Panel::Deployments => {
                if !app.deployments.is_empty() {
                    app.deploy_index = (app.deploy_index + 1).min(app.deployments.len() - 1);
                }
            }
        },
        KeyCode::Enter => match app.panel {
            Panel::Nodes if !app.nodes.is_empty() => app.view = View::NodeDetail,
            Panel::Deployments if !app.deployments.is_empty() => {
                app.view = View::DeploymentDetail
            }
            _ => {}
        },
        // Quick actions from dashboard
        KeyCode::Char('x') if app.panel == Panel::Nodes => {
            if let Some(node) = app.selected_node() {
                if node.status == NodeStatus::Offline {
                    let id = node.id;
                    let _ = client
                        .delete(format!("{}/api/nodes/{}", app.server, id))
                        .send()
                        .await;
                    app.set_status(format!("Removed node {}", &id.to_string()[..8]));
                    app.last_fetch = Instant::now() - Duration::from_secs(10);
                }
            }
        }
        KeyCode::Char('d') if app.panel == Panel::Deployments => {
            if let Some(dep) = app.selected_deployment() {
                let id = dep.id;
                let _ = client
                    .delete(format!("{}/api/deployments/{}", app.server, id))
                    .send()
                    .await;
                app.set_status(format!("Deleted deployment {}", &id.to_string()[..8]));
                app.last_fetch = Instant::now() - Duration::from_secs(10);
            }
        }
        KeyCode::Char('s') if app.panel == Panel::Deployments => {
            if let Some(dep) = app.selected_deployment() {
                if dep.status == DeploymentStatus::Running {
                    let id = dep.id;
                    let req = DeploymentControlRequest { action: DeploymentAction::Pause };
                    let _ = client
                        .put(format!("{}/api/deployments/{}", app.server, id))
                        .json(&req)
                        .send()
                        .await;
                    app.set_status(format!("Paused deployment {}", &id.to_string()[..8]));
                    app.last_fetch = Instant::now() - Duration::from_secs(10);
                }
            }
        }
        KeyCode::Char('r') if app.panel == Panel::Deployments => {
            if let Some(dep) = app.selected_deployment() {
                let id = dep.id;
                let action = if dep.status == DeploymentStatus::Paused {
                    DeploymentAction::Resume
                } else {
                    DeploymentAction::Restart
                };
                let req = DeploymentControlRequest { action };
                let _ = client
                    .put(format!("{}/api/deployments/{}", app.server, id))
                    .json(&req)
                    .send()
                    .await;
                let label = if action == DeploymentAction::Resume { "Resumed" } else { "Restarted" };
                app.set_status(format!("{} deployment {}", label, &id.to_string()[..8]));
                app.last_fetch = Instant::now() - Duration::from_secs(10);
            }
        }
        _ => {}
    }
}

async fn handle_node_detail_input(app: &mut App, client: &reqwest::Client, code: KeyCode) {
    match code {
        KeyCode::Char('x') => {
            if let Some(node) = app.selected_node() {
                if node.status == NodeStatus::Offline {
                    let id = node.id;
                    let _ = client
                        .delete(format!("{}/api/nodes/{}", app.server, id))
                        .send()
                        .await;
                    app.set_status(format!("Removed node {}", &id.to_string()[..8]));
                    app.view = View::Dashboard;
                    app.last_fetch = Instant::now() - Duration::from_secs(10);
                }
            }
        }
        KeyCode::Esc | KeyCode::Backspace => app.view = View::Dashboard,
        _ => {}
    }
}

async fn handle_deployment_detail_input(
    app: &mut App,
    client: &reqwest::Client,
    code: KeyCode,
) {
    match code {
        KeyCode::Char('s') => {
            if let Some(dep) = app.selected_deployment() {
                if dep.status == DeploymentStatus::Running {
                    let id = dep.id;
                    let req = DeploymentControlRequest { action: DeploymentAction::Pause };
                    let _ = client
                        .put(format!("{}/api/deployments/{}", app.server, id))
                        .json(&req)
                        .send()
                        .await;
                    app.set_status("Paused");
                    app.last_fetch = Instant::now() - Duration::from_secs(10);
                }
            }
        }
        KeyCode::Char('r') => {
            if let Some(dep) = app.selected_deployment() {
                let id = dep.id;
                let action = if dep.status == DeploymentStatus::Paused {
                    DeploymentAction::Resume
                } else {
                    DeploymentAction::Restart
                };
                let req = DeploymentControlRequest { action };
                let _ = client
                    .put(format!("{}/api/deployments/{}", app.server, id))
                    .json(&req)
                    .send()
                    .await;
                let label = if action == DeploymentAction::Resume { "Resumed" } else { "Restarted" };
                app.set_status(label);
                app.last_fetch = Instant::now() - Duration::from_secs(10);
            }
        }
        KeyCode::Char('d') => {
            if let Some(dep) = app.selected_deployment() {
                let id = dep.id;
                let _ = client
                    .delete(format!("{}/api/deployments/{}", app.server, id))
                    .send()
                    .await;
                app.set_status("Deleted");
                app.view = View::Dashboard;
                app.last_fetch = Instant::now() - Duration::from_secs(10);
            }
        }
        KeyCode::Esc | KeyCode::Backspace => app.view = View::Dashboard,
        _ => {}
    }
}

// ── Drawing ─────────────────────────────────────────────────────

fn draw(frame: &mut Frame, app: &App) {
    if app.show_help {
        draw_help(frame);
        return;
    }

    match app.view {
        View::Dashboard => draw_dashboard(frame, app),
        View::NodeDetail => draw_node_detail(frame, app),
        View::DeploymentDetail => draw_deployment_detail(frame, app),
    }
}

fn draw_dashboard(frame: &mut Frame, app: &App) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // title bar
            Constraint::Min(5),    // nodes
            Constraint::Min(5),    // deployments
            Constraint::Length(2), // footer
        ])
        .split(area);

    // Title bar
    draw_title_bar(frame, chunks[0], app);

    // Nodes panel
    draw_nodes_panel(frame, chunks[1], app);

    // Deployments panel
    draw_deployments_panel(frame, chunks[2], app);

    // Footer
    draw_footer(frame, chunks[3], app);
}

fn draw_title_bar(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" llamactl ")
        .title_alignment(Alignment::Center);

    let content = if let Some(ref err) = app.error_msg {
        Line::from(vec![
            Span::styled(" ERROR ", Style::default().fg(Color::Red).bold()),
            Span::raw(err.as_str()),
        ])
    } else if let Some((ref msg, _)) = app.status_msg {
        Line::from(Span::styled(
            format!(" {msg}"),
            Style::default().fg(Color::Green).bold(),
        ))
    } else {
        let now = chrono::Local::now().format("%H:%M:%S");
        let online = app.nodes.iter().filter(|n| n.status == NodeStatus::Online).count();
        let running = app
            .deployments
            .iter()
            .filter(|d| d.status == DeploymentStatus::Running)
            .count();
        Line::from(vec![
            Span::styled(
                format!(" {online}/{} nodes ", app.nodes.len()),
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                format!("{running}/{} deployments ", app.deployments.len()),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(format!(" {now}"), Style::default().fg(Color::DarkGray)),
        ])
    };

    frame.render_widget(Paragraph::new(content).block(block), area);
}

fn draw_nodes_panel(frame: &mut Frame, area: Rect, app: &App) {
    let is_active = app.panel == Panel::Nodes;
    let border_color = if is_active { Color::Cyan } else { Color::DarkGray };

    let header = Row::new(vec![
        "ID", "HOSTNAME", "STATUS", "BACKEND", "VRAM", "RAM", "CPU", "CACHED",
    ])
    .style(Style::default().fg(Color::Cyan).bold());

    let rows: Vec<Row> = app
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let status_style = match n.status {
                NodeStatus::Online => Style::default().fg(Color::Green),
                NodeStatus::Offline => Style::default().fg(Color::Red),
                NodeStatus::Draining => Style::default().fg(Color::Yellow),
            };

            let vram = match (n.vram_used_mb, n.vram_mb) {
                (Some(u), Some(t)) => format!("{}/{} MB", u, t),
                (None, Some(t)) => format!("{} MB", t),
                _ => "-".into(),
            };

            let ram_pct = if n.total_ram_mb > 0 {
                (n.ram_used_mb as f64 / n.total_ram_mb as f64 * 100.0) as u32
            } else {
                0
            };

            let cpu_bar = make_bar(n.cpu_usage_pct as f64, 100.0);
            let cached = if n.cached_models.is_empty() {
                "-".into()
            } else {
                format!("{}", n.cached_models.len())
            };

            let style = if is_active && i == app.node_index {
                Style::default().bg(Color::DarkGray).fg(Color::White)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(n.id.to_string()),
                Cell::from(short_hostname(&n.hostname)),
                Cell::from(n.status.to_string()).style(status_style),
                Cell::from(n.backend.to_string()),
                Cell::from(vram),
                Cell::from(format!("{}% of {} MB", ram_pct, n.total_ram_mb)),
                Cell::from(format!("{} {:.0}%", cpu_bar, n.cpu_usage_pct)),
                Cell::from(cached),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(36),
            Constraint::Length(16),
            Constraint::Length(10),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Length(16),
            Constraint::Length(14),
            Constraint::Min(6),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .title(format!(
                " Nodes [{}] ",
                if is_active { "active" } else { "tab" }
            ))
            .title_style(Style::default().fg(Color::White).bold()),
    );

    frame.render_widget(table, area);
}

fn draw_deployments_panel(frame: &mut Frame, area: Rect, app: &App) {
    let is_active = app.panel == Panel::Deployments;
    let border_color = if is_active { Color::Cyan } else { Color::DarkGray };

    let header = Row::new(vec!["ID", "MODEL", "NODE", "STATUS", "PORT", "ENDPOINT"])
        .style(Style::default().fg(Color::Cyan).bold());

    let rows: Vec<Row> = app
        .deployments
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let status_style = match d.status {
                DeploymentStatus::Running => Style::default().fg(Color::Green),
                DeploymentStatus::Pending | DeploymentStatus::Starting => {
                    Style::default().fg(Color::Yellow)
                }
                DeploymentStatus::Loading | DeploymentStatus::Transferring | DeploymentStatus::WarmingUp => {
                    Style::default().fg(Color::Cyan)
                }
                DeploymentStatus::Paused => Style::default().fg(Color::Blue),
                DeploymentStatus::Failed => Style::default().fg(Color::Red),
                DeploymentStatus::Stopped => Style::default().fg(Color::DarkGray),
            };

            let model = short_model(&d.model);

            let style = if is_active && i == app.deploy_index {
                Style::default().bg(Color::DarkGray).fg(Color::White)
            } else {
                Style::default()
            };

            // Rich status with progress
            let status_display = match d.status {
                DeploymentStatus::Loading | DeploymentStatus::Transferring | DeploymentStatus::WarmingUp | DeploymentStatus::Starting => {
                    format!("{} {}%", d.status, d.progress as u32)
                }
                DeploymentStatus::Running => {
                    if let Some(tps) = d.last_tok_per_sec {
                        format!("running {:.0} t/s", tps)
                    } else {
                        "running".into()
                    }
                }
                _ => d.status.to_string(),
            };

            // Show workers count for split deployments
            let node_display = if d.workers.is_empty() {
                short_hostname(&d.node_hostname)
            } else {
                format!("{} +{}", short_hostname(&d.node_hostname), d.workers.len() - 1)
            };

            Row::new(vec![
                Cell::from(d.id.to_string()),
                Cell::from(model),
                Cell::from(node_display),
                Cell::from(status_display).style(status_style),
                Cell::from(d.port.to_string()),
                Cell::from(d.endpoint.as_deref().unwrap_or("-").to_string()),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(36),
            Constraint::Length(28),
            Constraint::Length(18),
            Constraint::Length(18),
            Constraint::Length(6),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .title(format!(
                " Deployments [{}] ",
                if is_active { "active" } else { "tab" }
            ))
            .title_style(Style::default().fg(Color::White).bold()),
    );

    frame.render_widget(table, area);
}

fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let keys = match (app.view, app.panel) {
        (View::Dashboard, Panel::Nodes) => vec![
            ("tab", "switch panel"),
            ("j/k", "navigate"),
            ("enter", "details"),
            ("x", "remove offline"),
            ("?", "help"),
            ("q", "quit"),
        ],
        (View::Dashboard, Panel::Deployments) => vec![
            ("tab", "switch panel"),
            ("j/k", "navigate"),
            ("enter", "details"),
            ("s", "pause"),
            ("r", "resume/restart"),
            ("d", "delete"),
            ("q", "quit"),
        ],
        _ => vec![("esc", "back"), ("?", "help")],
    };

    let spans: Vec<Span> = keys
        .iter()
        .flat_map(|(key, desc)| {
            vec![
                Span::styled(format!(" {key} "), Style::default().fg(Color::Yellow).bold()),
                Span::styled(format!("{desc} "), Style::default().fg(Color::DarkGray)),
            ]
        })
        .collect();

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_node_detail(frame: &mut Frame, app: &App) {
    let node = match app.selected_node() {
        Some(n) => n,
        None => {
            frame.render_widget(Paragraph::new("No node selected"), frame.area());
            return;
        }
    };

    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12), // info
            Constraint::Min(4),    // cached models
            Constraint::Length(2), // footer
        ])
        .split(area);

    // Node info
    let status_color = match node.status {
        NodeStatus::Online => Color::Green,
        NodeStatus::Offline => Color::Red,
        NodeStatus::Draining => Color::Yellow,
    };

    let vram = match (node.vram_used_mb, node.vram_mb) {
        (Some(u), Some(t)) => format!("{u}/{t} MB"),
        (None, Some(t)) => format!("{t} MB"),
        _ => "-".into(),
    };

    let last_seen = format_duration_since(node.last_heartbeat);

    let info_lines = vec![
        Line::from(vec![
            Span::styled(" Hostname: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&node.hostname, Style::default().bold()),
        ]),
        Line::from(vec![
            Span::styled(" ID:       ", Style::default().fg(Color::DarkGray)),
            Span::raw(node.id.to_string()),
        ]),
        Line::from(vec![
            Span::styled(" Status:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(node.status.to_string(), Style::default().fg(status_color)),
            Span::styled(format!("  (last seen {last_seen})"), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled(" Backend:  ", Style::default().fg(Color::DarkGray)),
            Span::raw(node.backend.to_string()),
        ]),
        Line::from(vec![
            Span::styled(" OS:       ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{} ({})", node.os, node.arch)),
        ]),
        Line::from(vec![
            Span::styled(" VRAM:     ", Style::default().fg(Color::DarkGray)),
            Span::raw(vram),
        ]),
        Line::from(vec![
            Span::styled(" RAM:      ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{}/{} MB", node.ram_used_mb, node.total_ram_mb)),
        ]),
        Line::from(vec![
            Span::styled(" CPU:      ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!(
                "{} cores, {:.1}%  {}",
                node.cpu_cores,
                node.cpu_usage_pct,
                make_bar(node.cpu_usage_pct as f64, 100.0)
            )),
        ]),
    ];

    let info = Paragraph::new(info_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(format!(" {} ", short_hostname(&node.hostname)))
            .title_style(Style::default().fg(Color::White).bold()),
    );
    frame.render_widget(info, chunks[0]);

    // Cached models
    let model_lines: Vec<Line> = if node.cached_models.is_empty() {
        vec![Line::from(Span::styled(
            " No cached models",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        node.cached_models
            .iter()
            .map(|m| {
                Line::from(vec![
                    Span::styled(" - ", Style::default().fg(Color::DarkGray)),
                    Span::raw(m.as_str()),
                ])
            })
            .collect()
    };

    let models = Paragraph::new(model_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(format!(" Cached Models ({}) ", node.cached_models.len()))
            .title_style(Style::default().fg(Color::White).bold()),
    );
    frame.render_widget(models, chunks[1]);

    // Footer
    let mut keys = vec![("esc", "back")];
    if node.status == NodeStatus::Offline {
        keys.push(("x", "remove node"));
    }
    let spans: Vec<Span> = keys
        .iter()
        .flat_map(|(key, desc)| {
            vec![
                Span::styled(format!(" {key} "), Style::default().fg(Color::Yellow).bold()),
                Span::styled(format!("{desc} "), Style::default().fg(Color::DarkGray)),
            ]
        })
        .collect();
    frame.render_widget(Paragraph::new(Line::from(spans)), chunks[2]);
}

fn draw_deployment_detail(frame: &mut Frame, app: &App) {
    let dep = match app.selected_deployment() {
        Some(d) => d,
        None => {
            frame.render_widget(Paragraph::new("No deployment selected"), frame.area());
            return;
        }
    };

    let area = frame.area();
    let has_workers = !dep.workers.is_empty();
    let _has_logs = !dep.logs.is_empty();

    let constraints = if has_workers {
        vec![
            Constraint::Length(12),                              // info + progress
            Constraint::Length((dep.workers.len() as u16) + 4), // topology
            Constraint::Min(6),                                 // logs
            Constraint::Length(2),                               // footer
        ]
    } else {
        vec![
            Constraint::Length(12), // info + progress
            Constraint::Min(6),    // logs
            Constraint::Length(2), // footer
        ]
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let status_color = match dep.status {
        DeploymentStatus::Running => Color::Green,
        DeploymentStatus::Pending | DeploymentStatus::Starting => Color::Yellow,
        DeploymentStatus::Loading | DeploymentStatus::Transferring | DeploymentStatus::WarmingUp => Color::Cyan,
        DeploymentStatus::Paused => Color::Blue,
        DeploymentStatus::Failed => Color::Red,
        DeploymentStatus::Stopped => Color::DarkGray,
    };

    let age = format_duration_since(dep.created_at);

    // Status line with progress bar
    let status_line = match dep.status {
        DeploymentStatus::Loading | DeploymentStatus::Transferring | DeploymentStatus::WarmingUp | DeploymentStatus::Starting => {
            let bar = make_progress_bar(dep.progress as f64, 100.0, 20);
            vec![
                Span::styled(dep.status.to_string(), Style::default().fg(status_color).bold()),
                Span::raw("  "),
                Span::styled(bar, Style::default().fg(Color::Cyan)),
                Span::raw(format!(" {:.0}%", dep.progress)),
            ]
        }
        DeploymentStatus::Running => {
            let mut spans = vec![
                Span::styled("running", Style::default().fg(Color::Green).bold()),
            ];
            if let Some(tps) = dep.last_tok_per_sec {
                spans.push(Span::raw("  "));
                spans.push(Span::styled(format!("{:.1} tok/s", tps), Style::default().fg(Color::Yellow).bold()));
            }
            spans
        }
        _ => vec![Span::styled(dep.status.to_string(), Style::default().fg(status_color).bold())],
    };

    let model_short = short_model(&dep.model);
    let mut info_lines = vec![
        Line::from(vec![
            Span::styled(" Model:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(model_short, Style::default().bold()),
        ]),
        Line::from(vec![
            Span::styled(" Full path: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&dep.model, Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled(" ID:        ", Style::default().fg(Color::DarkGray)),
            Span::raw(dep.id.to_string()),
        ]),
        Line::from(vec![
            Span::styled(" Node:      ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{} ({})", short_hostname(&dep.node_hostname), &dep.node_id.to_string()[..8])),
        ]),
        Line::from(std::iter::once(Span::styled(" Status:    ", Style::default().fg(Color::DarkGray))).chain(status_line).collect::<Vec<_>>()),
        Line::from(vec![
            Span::styled(" Endpoint:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                dep.endpoint.as_deref().unwrap_or("-"),
                if dep.endpoint.is_some() { Style::default().fg(Color::Green) } else { Style::default().fg(Color::DarkGray) },
            ),
        ]),
        Line::from(vec![
            Span::styled(" Config:    ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("ctx={}, gpu_layers={}, port={}", dep.context_length, dep.gpu_layers, dep.port)),
        ]),
        Line::from(vec![
            Span::styled(" Created:   ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{age} ago")),
        ]),
    ];

    if let Some(ref err) = dep.error {
        info_lines.push(Line::from(vec![
            Span::styled(" Error:     ", Style::default().fg(Color::Red)),
            Span::styled(err.as_str(), Style::default().fg(Color::Red)),
        ]));
    }

    let info = Paragraph::new(info_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(format!(" Deployment {} ", &dep.id.to_string()[..8]))
            .title_style(Style::default().fg(Color::White).bold()),
    );
    frame.render_widget(info, chunks[0]);

    let mut next_chunk = 1;

    // Topology section for split deployments
    if has_workers {
        let mut topo_lines: Vec<Line> = Vec::new();
        for w in &dep.workers {
            let role_icon = match w.role {
                AssignmentRole::Main => "main",
                AssignmentRole::RpcWorker => " rpc",
            };
            let role_color = match w.role {
                AssignmentRole::Main => Color::Green,
                AssignmentRole::RpcWorker => Color::Yellow,
            };
            let buf_info = w.buffer_size_mb
                .map(|mb| format!(" ({:.0} MB)", mb))
                .unwrap_or_default();

            topo_lines.push(Line::from(vec![
                Span::styled(format!("  [{role_icon}] "), Style::default().fg(role_color).bold()),
                Span::raw(format!("{}", short_hostname(&w.hostname))),
                Span::styled(format!(" :{}", w.rpc_port), Style::default().fg(Color::DarkGray)),
                Span::styled(buf_info, Style::default().fg(Color::Cyan)),
            ]));
        }

        let topo = Paragraph::new(topo_lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(format!(" Split Topology ({} nodes) ", dep.workers.len()))
                .title_style(Style::default().fg(Color::White).bold()),
        );
        frame.render_widget(topo, chunks[next_chunk]);
        next_chunk += 1;
    }

    // Logs section
    let log_lines: Vec<Line> = if dep.logs.is_empty() && dep.error.is_none() {
        vec![Line::from(Span::styled(
            " Waiting for output...",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        dep.logs
            .iter()
            .map(|l| {
                let style = if l.contains("error") || l.contains("failed") || l.contains("ABORT") {
                    Style::default().fg(Color::Red)
                } else if l.contains("warning") || l.contains("WARNING") {
                    Style::default().fg(Color::Yellow)
                } else if l.contains("model loaded") || l.contains("server is listening") {
                    Style::default().fg(Color::Green)
                } else if l.contains("RPC") || l.contains("rpc") {
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                Line::from(Span::styled(format!(" {l}"), style))
            })
            .collect()
    };

    // Auto-scroll to bottom
    let visible_height = chunks[next_chunk].height.saturating_sub(2) as usize;
    let scroll = if log_lines.len() > visible_height {
        (log_lines.len() - visible_height) as u16
    } else {
        0
    };

    let logs_widget = Paragraph::new(log_lines)
        .scroll((scroll, 0))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(format!(" Logs ({}) ", dep.logs.len()))
                .title_style(Style::default().fg(Color::White).bold()),
        );
    frame.render_widget(logs_widget, chunks[next_chunk]);
    next_chunk += 1;

    // Footer
    let mut keys: Vec<(&str, &str)> = vec![("esc", "back")];
    match dep.status {
        DeploymentStatus::Running => {
            keys.push(("s", "pause"));
            keys.push(("r", "restart"));
        }
        DeploymentStatus::Paused => {
            keys.push(("r", "resume"));
        }
        DeploymentStatus::Failed | DeploymentStatus::Stopped => {
            keys.push(("r", "restart"));
        }
        _ => {}
    }
    keys.push(("d", "delete"));

    let spans: Vec<Span> = keys
        .iter()
        .flat_map(|(key, desc)| {
            vec![
                Span::styled(format!(" {key} "), Style::default().fg(Color::Yellow).bold()),
                Span::styled(format!("{desc} "), Style::default().fg(Color::DarkGray)),
            ]
        })
        .collect();
    frame.render_widget(Paragraph::new(Line::from(spans)), chunks[next_chunk]);
}

fn draw_help(frame: &mut Frame) {
    let area = centered_rect(60, 70, frame.area());

    let help_text = vec![
        Line::from(Span::styled(
            "Keyboard Shortcuts",
            Style::default().bold().fg(Color::Cyan),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Dashboard", Style::default().bold()),
        ]),
        Line::from("  tab        Switch between Nodes / Deployments"),
        Line::from("  j/k ↑/↓    Navigate up/down"),
        Line::from("  enter      Open detail view"),
        Line::from("  q          Quit"),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Nodes Panel", Style::default().bold()),
        ]),
        Line::from("  x          Remove offline node"),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Deployments Panel", Style::default().bold()),
        ]),
        Line::from("  s          Pause running deployment"),
        Line::from("  r          Resume paused / restart failed"),
        Line::from("  d          Delete deployment"),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Detail Views", Style::default().bold()),
        ]),
        Line::from("  esc        Back to dashboard"),
        Line::from("  Same keys as panel for actions"),
        Line::from(""),
        Line::from(Span::styled(
            "  Press ? or esc to close",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Help ")
        .title_alignment(Alignment::Center);

    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(help_text).block(block),
        area,
    );
}

// ── Helpers ─────────────────────────────────────────────────────

fn make_bar(value: f64, max: f64) -> String {
    let width = 8;
    let filled = ((value / max) * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn make_progress_bar(value: f64, max: f64, width: usize) -> String {
    let filled = ((value / max) * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

fn short_hostname(hostname: &str) -> String {
    // Strip .local, .lan, .home suffixes
    hostname
        .strip_suffix(".local")
        .or_else(|| hostname.strip_suffix(".lan"))
        .or_else(|| hostname.strip_suffix(".home"))
        .unwrap_or(hostname)
        .to_string()
}

fn short_model(model: &str) -> String {
    // Show just the filename or last path component
    let name = model.rsplit('/').next().unwrap_or(model);
    if name.len() > 30 {
        format!("{}...", &name[..27])
    } else {
        name.to_string()
    }
}

fn format_duration_since(dt: chrono::DateTime<chrono::Utc>) -> String {
    let now = chrono::Utc::now();
    let dur = now - dt;

    if dur.num_days() > 0 {
        format!("{}d {}h", dur.num_days(), dur.num_hours() % 24)
    } else if dur.num_hours() > 0 {
        format!("{}h {}m", dur.num_hours(), dur.num_minutes() % 60)
    } else if dur.num_minutes() > 0 {
        format!("{}m {}s", dur.num_minutes(), dur.num_seconds() % 60)
    } else {
        format!("{}s", dur.num_seconds())
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

async fn fetch_data(
    client: &reqwest::Client,
    server: &str,
) -> Result<(Vec<NodeInfo>, Vec<DeploymentInfo>)> {
    let (nodes_resp, deploy_resp) = tokio::join!(
        client.get(format!("{server}/api/nodes")).send(),
        client.get(format!("{server}/api/deployments")).send(),
    );

    let nodes: Vec<NodeInfo> = nodes_resp?.error_for_status()?.json().await?;
    let deployments: Vec<DeploymentInfo> = deploy_resp?.error_for_status()?.json().await?;

    Ok((nodes, deployments))
}
