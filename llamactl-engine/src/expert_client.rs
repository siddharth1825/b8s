//! gRPC client for remote expert computation.
//! Used by the fast GPU device to offload expert FFN to the big memory device.

use std::time::Instant;
use anyhow::Result;
use candle_core::{DType, Tensor};
use tonic::transport::Channel;

use crate::rpc_proto::expert_compute_client::ExpertComputeClient;
use crate::rpc_proto::LayerExpertRequest;
use crate::expert_server::{tensor_to_bytes, bytes_to_tensor};

/// Per-hop timing info
#[derive(Debug, Clone)]
pub struct HopTiming {
    pub layer: u32,
    pub serialize_us: u64,
    pub network_us: u64,
    pub remote_compute_us: u64,
    pub deserialize_us: u64,
    pub total_us: u64,
}

pub struct RemoteExpertClient {
    client: ExpertComputeClient<Channel>,
    pub timings: Vec<HopTiming>,
}

impl RemoteExpertClient {
    pub async fn connect(addr: &str) -> Result<Self> {
        let client = ExpertComputeClient::connect(addr.to_string()).await?;
        Ok(Self {
            client,
            timings: Vec::new(),
        })
    }

    /// Compute expert + shared FFN for a single layer on the remote device.
    /// Returns the combined output tensor [batch, seq_len, hidden].
    pub async fn compute_layer(
        &mut self,
        layer: u32,
        hidden_state: &Tensor,
        expert_indices: &Tensor,
        expert_weights: &Tensor,
        k: u32,
    ) -> Result<Tensor> {
        let total_start = Instant::now();

        let (batch, seq_len, hidden_dim) = hidden_state.dims3()?;

        // Serialize tensors
        let ser_start = Instant::now();
        let hidden_bytes = tensor_to_bytes(hidden_state)?;
        let indices_flat = expert_indices.to_device(&candle_core::Device::Cpu)?.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
        let weights_flat = expert_weights.to_device(&candle_core::Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let serialize_us = ser_start.elapsed().as_micros() as u64;

        // gRPC call
        let net_start = Instant::now();
        let request = LayerExpertRequest {
            layer,
            hidden_state: hidden_bytes,
            batch_size: batch as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            expert_indices: indices_flat,
            expert_weights: weights_flat,
            k,
        };

        let response = self.client.compute_layer(request).await?.into_inner();
        let network_us = net_start.elapsed().as_micros() as u64;

        // Deserialize output
        let deser_start = Instant::now();
        let output = bytes_to_tensor(&response.output, batch as u32, seq_len as u32, hidden_dim as u32)?;
        let deserialize_us = deser_start.elapsed().as_micros() as u64;

        let total_us = total_start.elapsed().as_micros() as u64;

        // Log timing
        let timing = HopTiming {
            layer,
            serialize_us,
            network_us: network_us.saturating_sub(response.compute_us),
            remote_compute_us: response.compute_us,
            deserialize_us,
            total_us,
        };

        tracing::debug!(
            layer = layer,
            serialize_us = timing.serialize_us,
            network_us = timing.network_us,
            compute_us = timing.remote_compute_us,
            deserialize_us = timing.deserialize_us,
            total_us = timing.total_us,
            "Expert hop"
        );

        self.timings.push(timing);

        Ok(output)
    }

    /// Print timing summary after a full forward pass
    pub fn timing_summary(&self) -> String {
        if self.timings.is_empty() {
            return "No timings recorded".into();
        }

        let n = self.timings.len();
        let total: u64 = self.timings.iter().map(|t| t.total_us).sum();
        let serialize: u64 = self.timings.iter().map(|t| t.serialize_us).sum();
        let network: u64 = self.timings.iter().map(|t| t.network_us).sum();
        let compute: u64 = self.timings.iter().map(|t| t.remote_compute_us).sum();
        let deserialize: u64 = self.timings.iter().map(|t| t.deserialize_us).sum();

        format!(
            "{} hops, {:.1}ms total\n  serialize: {:.1}ms ({:.0}%), network: {:.1}ms ({:.0}%), compute: {:.1}ms ({:.0}%), deserialize: {:.1}ms ({:.0}%)",
            n,
            total as f64 / 1000.0,
            serialize as f64 / 1000.0, serialize as f64 / total as f64 * 100.0,
            network as f64 / 1000.0, network as f64 / total as f64 * 100.0,
            compute as f64 / 1000.0, compute as f64 / total as f64 * 100.0,
            deserialize as f64 / 1000.0, deserialize as f64 / total as f64 * 100.0,
        )
    }

    pub fn clear_timings(&mut self) {
        self.timings.clear();
    }
}
