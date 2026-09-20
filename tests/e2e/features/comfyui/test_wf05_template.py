# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Portable graph and temporal-mask contracts for the WF-05 examples."""

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WORKFLOW = (
    Path(__file__).resolve().parents[4] / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni H3 Latent Editing.json"
)


def test_wf05_graph_links():
    graph = json.loads(WORKFLOW.read_text())
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert len(nodes) == len(graph["nodes"])
    assert len({link[0] for link in graph["links"]}) == len(graph["links"])
    for link_id, source, output, target, slot, kind in graph["links"]:
        assert nodes[target]["inputs"][slot]["link"] == link_id
        assert link_id in nodes[source]["outputs"][output]["links"]
        assert nodes[source]["outputs"][output]["type"] == kind
        assert nodes[target]["inputs"][slot]["type"] == kind
        assert nodes[source]["mode"] == nodes[target]["mode"]


def test_wf05_defaults_and_outputs():
    graph = json.loads(WORKFLOW.read_text())
    generators = [n for n in graph["nodes"] if n["type"] == "VLLMOmniGenerateVideo"]
    assert len(generators) == 4
    assert [n["mode"] for n in generators] == [0, 2, 2, 2]
    for node in generators:
        width, height, fps, duration = node["widgets_values"][-4:]
        assert (width, height, fps) == (1344, 768, 24)
        frames = round(duration * fps)
        assert frames % 17 == 5
        assert frames == (209 if node["title"] == "Extension" else 107)
        assert next(i for i in node["inputs"] if i["name"] == "latent_edit")["link"] is not None
    saves = [n for n in graph["nodes"] if n["type"] == "SaveVideo"]
    assert len(saves) == 4
    assert all(n["inputs"][0]["link"] is not None for n in saves)


@pytest.mark.parametrize("title,counts", [("Continuation", (16, 16)), ("Extension", (32, 30))])
def test_wf05_temporal_mask(title, counts):
    graph = json.loads(WORKFLOW.read_text())
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}

    def upstream(node, name):
        link_id = next(i["link"] for i in node["inputs"] if i["name"] == name)
        return nodes[links[link_id][1]]

    generator = next(n for n in nodes.values() if n["title"] == title)
    edit = upstream(generator, "latent_edit")
    assert edit["widgets_values"] == [1.0]
    mask = upstream(edit, "video_mask")
    assert mask["type"] == "ImageToMask"
    batch = upstream(mask, "image")
    assert batch["type"] == "ImageBatch"
    for slot, value, count in zip(("image1", "image2"), (0.0, 1.0), counts):
        repeat = upstream(batch, slot)
        assert repeat["type"] == "RepeatImageBatch"
        assert repeat["widgets_values"] == [count]
        solid = upstream(upstream(repeat, "image"), "mask")
        assert solid["type"] == "SolidMask"
        assert solid["widgets_values"][0] == value
    frames = round(generator["widgets_values"][-1] * 24)
    assert sum(counts) == 2 + 5 * ((frames - 5) // 17)
