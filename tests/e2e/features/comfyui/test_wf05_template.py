# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Portable graph and temporal-mask contracts for the WF-05 examples."""

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WORKFLOW = (
    Path(__file__).resolve().parents[4]
    / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Latent Mask Editing.json"
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
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    generators = [n for n in nodes.values() if n["type"] == "VLLMOmniGenerateVideo"]
    assert len(generators) == 4
    assert len([n for n in nodes.values() if n["type"] == "SaveVideo"]) == 8
    assert not any(n["type"] == "VLLMOmniH3MaskGridPreview" for n in nodes.values())
    for gen in generators:
        case = gen["title"].removeprefix("Generate Video(").removesuffix(")")
        save = next(n for n in nodes.values() if n.get("title") == f"Mask Preview ({case})")
        create = nodes[links[save["inputs"][0]["link"]][1]]
        image_node = nodes[links[create["inputs"][0]["link"]][1]]
        edit = nodes[links[next(i["link"] for i in gen["inputs"] if i["name"] == "latent_edit")][1]]
        mask_link = next(i["link"] for i in edit["inputs"] if i["name"] == "video_mask")
        assert image_node["type"] == "ImageBlend"
        consumer = nodes[links[next(i["link"] for i in image_node["inputs"] if i["name"] == "image2")][1]]
        assert consumer["type"] == "ImageCompositeMasked"
        preview_mask = next(i["link"] for i in consumer["inputs"] if i["name"] == "mask")
        if case in ("Continuation", "Extension"):
            assert nodes[links[mask_link][1]]["type"] == "VLLMOmniH3TemporalMask"
            assert links[preview_mask][1:3] == [links[mask_link][1], 4]
            fps_link = next(i["link"] for i in create["inputs"] if i["name"] == "fps")
            assert links[fps_link][1:3] == [links[mask_link][1], 2]
        else:
            assert links[mask_link][1:3] == links[preview_mask][1:3]


def test_wf05_output_layout():
    graph = json.loads(WORKFLOW.read_text())
    for case in ("Object Removal", "Inpainting", "Continuation", "Extension"):
        preview = next(n for n in graph["nodes"] if n.get("title") == f"Mask Preview ({case})")
        result = next(n for n in graph["nodes"] if n.get("title") == f"Save Video ({case})")
        assert preview["pos"][0] == result["pos"][0]
        assert preview["pos"][1] + preview["size"][1] < result["pos"][1]


def test_wf05_portable_inputs():
    graph = json.loads(WORKFLOW.read_text())
    for node in graph["nodes"]:
        if node["type"] in ("LoadVideo", "LoadImageMask"):
            assert node["widgets_values"][0] == ""
    assert not any(n["type"] == "ImageBatch" for n in graph["nodes"])


def test_wf05_temporal_nodes():
    graph = json.loads(WORKFLOW.read_text())
    assert sum(n["type"] == "VLLMOmniH3TemporalMask" for n in graph["nodes"]) == 2
    assert not any(n["type"] == "ComfyMathExpression" for n in graph["nodes"])
