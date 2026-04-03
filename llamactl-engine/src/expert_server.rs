//! Expert worker gRPC server — runs on the big memory device.
//! Holds all expert FFN weights and computes sparse expert + shared expert FFN.

use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status};

use candle_core::{DType, Tensor};

use crate::loader::LoadedModel;
use crate::rpc_proto::{
    expert_compute_server::ExpertCompute,
    ExpertRequest, ExpertResponse, LayerExpertRequest,
    PingRequest, PingResponse, SharedExpertRequest,
};

pub struct ExpertWorker {
    model: Arc<LoadedModel>,
}

impl ExpertWorker {
    pub fn new(model: Arc<LoadedModel>) -> Self {
        Self { model }
    }
}

#[tonic::async_trait]
impl ExpertCompute for ExpertWorker {
    async fn compute_experts(
        &self,
        request: Request<ExpertRequest>,
    ) -> Result<Response<ExpertResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let result = compute_expert_ffn(&self.model, &req)
            .map_err(|e| Status::internal(format!("Expert compute failed: {}", e)))?;

        let compute_us = start.elapsed().as_micros() as u64;

        Ok(Response::new(ExpertResponse {
            output: tensor_to_bytes(&result)
                .map_err(|e| Status::internal(format!("Serialize failed: {}", e)))?,
            compute_us,
            total_us: start.elapsed().as_micros() as u64,
        }))
    }

    async fn compute_shared_experts(
        &self,
        request: Request<SharedExpertRequest>,
    ) -> Result<Response<ExpertResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let hidden = bytes_to_tensor(&req.hidden_state, req.batch_size, req.seq_len, req.hidden_dim)
            .map_err(|e| Status::internal(format!("Deserialize failed: {}", e)))?;

        let result = crate::inference::shared_expert_ffn(&hidden, &self.model, req.layer)
            .map_err(|e| Status::internal(format!("Shared expert compute failed: {}", e)))?;

        let compute_us = start.elapsed().as_micros() as u64;

        Ok(Response::new(ExpertResponse {
            output: tensor_to_bytes(&result)
                .map_err(|e| Status::internal(format!("Serialize failed: {}", e)))?,
            compute_us,
            total_us: start.elapsed().as_micros() as u64,
        }))
    }

    async fn compute_layer(
        &self,
        request: Request<LayerExpertRequest>,
    ) -> Result<Response<ExpertResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let hidden = bytes_to_tensor(&req.hidden_state, req.batch_size, req.seq_len, req.hidden_dim)
            .map_err(|e| Status::internal(format!("Deserialize failed: {}", e)))?;

        let device = candle_core::Device::Cpu;
        let k = req.k as usize;
        let batch = req.batch_size as usize;
        let seq_len = req.seq_len as usize;
        let n_tokens = batch * seq_len;

        // Reconstruct expert indices and weights tensors
        let indices = Tensor::from_vec(
            req.expert_indices.iter().map(|&x| x).collect::<Vec<u32>>(),
            (batch, seq_len, k),
            &device,
        ).map_err(|e| Status::internal(e.to_string()))?;

        let weights = Tensor::from_vec(
            req.expert_weights.clone(),
            (batch, seq_len, k),
            &device,
        ).map_err(|e| Status::internal(e.to_string()))?;

        // Compute expert FFN
        let expert_out = crate::inference::expert_ffn(
            &hidden, &indices, &weights, &self.model, req.layer,
        ).map_err(|e| Status::internal(format!("Expert FFN failed: {}", e)))?;

        // Compute shared expert FFN
        let shared_out = crate::inference::shared_expert_ffn(&hidden, &self.model, req.layer)
            .map_err(|e| Status::internal(format!("Shared FFN failed: {}", e)))?;

        // Combine
        let combined = (&expert_out + &shared_out)
            .map_err(|e| Status::internal(e.to_string()))?;

        let compute_us = start.elapsed().as_micros() as u64;

        Ok(Response::new(ExpertResponse {
            output: tensor_to_bytes(&combined)
                .map_err(|e| Status::internal(e.to_string()))?,
            compute_us,
            total_us: start.elapsed().as_micros() as u64,
        }))
    }

    async fn ping(
        &self,
        _request: Request<PingRequest>,
    ) -> Result<Response<PingResponse>, Status> {
        let expert_mb = self.model.big_memory.values()
            .map(|t| t.storage_size_in_bytes() as u64)
            .sum::<u64>() / (1024 * 1024);

        Ok(Response::new(PingResponse {
            device_name: "expert-worker".into(),
            n_layers: self.model.n_layers,
            expert_memory_mb: expert_mb,
        }))
    }
}

// ── Helper: compute expert FFN from protobuf request ────────────

fn compute_expert_ffn(model: &LoadedModel, req: &ExpertRequest) -> anyhow::Result<Tensor> {
    let device = candle_core::Device::Cpu;
    let k = req.k as usize;
    let batch = req.batch_size as usize;
    let seq_len = req.seq_len as usize;

    let hidden = bytes_to_tensor(&req.hidden_state, req.batch_size, req.seq_len, req.hidden_dim)?;

    let indices = Tensor::from_vec(
        req.expert_indices.iter().map(|&x| x).collect::<Vec<u32>>(),
        (batch, seq_len, k),
        &device,
    )?;

    let weights = Tensor::from_vec(
        req.expert_weights.clone(),
        (batch, seq_len, k),
        &device,
    )?;

    crate::inference::expert_ffn(&hidden, &indices, &weights, model, req.layer)
}

// ── Tensor serialization ────────────────────────────────────────

pub fn tensor_to_bytes(t: &Tensor) -> anyhow::Result<Vec<u8>> {
    let t = t.to_device(&candle_core::Device::Cpu)?.to_dtype(DType::F32)?;
    let data = t.flatten_all()?.to_vec1::<f32>()?;
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Ok(bytes)
}

pub fn bytes_to_tensor(bytes: &[u8], batch: u32, seq_len: u32, hidden: u32) -> anyhow::Result<Tensor> {
    let n_floats = bytes.len() / 4;
    let data: Vec<f32> = (0..n_floats)
        .map(|i| f32::from_le_bytes([bytes[i*4], bytes[i*4+1], bytes[i*4+2], bytes[i*4+3]]))
        .collect();
    let t = Tensor::from_vec(data, (batch as usize, seq_len as usize, hidden as usize), &candle_core::Device::Cpu)?;
    Ok(t)
}
