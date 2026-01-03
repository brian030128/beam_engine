import torch
import flashinfer
from beam_engine.logger import init_logger

logger = init_logger(__name__)

DEVICE = "cuda:5"
num_layers = 32
num_qo_heads = 64
num_kv_heads = 8
head_dim = 128
page_size = 16
# allocate 128MB workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
    2, workspace_buffer, "NHD"
)
batch_size = 7
shared_kv_num_pages = 512
unique_kv_num_pages = 128
total_num_pages = shared_kv_num_pages + unique_kv_num_pages
shared_kv_page_indices = torch.arange(shared_kv_num_pages).int().to(DEVICE)
shared_kv_page_indptr = torch.tensor([0, shared_kv_num_pages], dtype=torch.int32, device=DEVICE)
unique_kv_page_indices = torch.arange(shared_kv_num_pages, total_num_pages).int().to(DEVICE)
unique_kv_page_indptr = torch.tensor(
    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device=DEVICE
)
shared_kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device=DEVICE)
# 1 <= kv_last_page_len <= page_size
unique_kv_last_page_len = torch.tensor(
    [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device=DEVICE
)
kv_cache_at_layer = [
    torch.randn(
        total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=DEVICE
    ) for _ in range(num_layers)
]
qo_indptr_arr = [
    torch.tensor([0, batch_size], dtype=torch.int32, device=DEVICE),  # top-level for shared KV-Cache
    torch.arange(batch_size + 1, dtype=torch.int32, device=DEVICE)    # bottom-level for unique KV-Cache
]
# create auxiliary data structures for batch decode attention
wrapper.plan(
    qo_indptr_arr,
    [shared_kv_page_indptr, unique_kv_page_indptr],
    [shared_kv_page_indices, unique_kv_page_indices],
    [shared_kv_last_page_len, unique_kv_last_page_len],
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
)
outputs = []
for i in range(num_layers):
    q = torch.randn(batch_size, num_qo_heads, head_dim).half().to(DEVICE)
    # compute batch decode attention, reuse auxiliary data structures for all layers
    o = wrapper.run(q, kv_cache_at_layer[i])
    outputs.append(o)

logger.info(outputs[0].shape)