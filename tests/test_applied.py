import numpy as np

from optimumai.diffusion import forward_diffusion, forward_diffusion_trace
from optimumai.embeddings import embedding_lookup, nearest_neighbors_trace
from optimumai.rag import RAGPipeline


def test_embedding_lookup_shape():
    emb = embedding_lookup(["cat", "sat", "mat"], dim=4)
    assert emb.shape == (3, 4)


def test_nearest_neighbors_returns_ranked_list():
    t = nearest_neighbors_trace("king", ["queen", "man", "woman", "apple"], k=2)
    assert isinstance(t.result, list)
    assert len(t.result) == 2


def test_rag_pipeline_returns_prompt_string():
    t = RAGPipeline.demo()
    assert isinstance(t.result, str)
    assert len(t) >= 4  # embed query, score docs, select, assemble ...


def test_rag_topk_respected():
    rag = RAGPipeline()
    prompt = rag.forward("how do neural networks learn?", k=2)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_forward_diffusion_shape_and_noise_grows():
    x0 = np.linspace(-1, 1, 6)
    x_t = forward_diffusion(x0)
    assert x_t.shape == x0.shape
    # the noised signal should differ from the clean one
    assert not np.allclose(x_t, x0)


def test_forward_diffusion_trace_steps():
    t = forward_diffusion_trace(np.linspace(-1, 1, 6), timesteps=10)
    assert len(t) >= 4


def test_rag_render_does_not_crash(capsys):
    RAGPipeline.demo().render("engineer")  # string result must render safely
    assert "RAG" in capsys.readouterr().out.upper()
