"""Static notebook checks only: never execute cells, datasets or training."""

import ast
import json
from pathlib import Path

from roboqc.industrial.dataset import FAMILIES
from roboqc.industrial.voice import understand


def test_stage_documents_and_pretrained_quickstart_parse():
    root = Path(__file__).resolve().parents[2] / "colab"
    notebooks = sorted(root.glob("0*.ipynb"))
    assert len(notebooks) == 11
    assert (root / "04a_try_yoloe_pretrained.ipynb") in notebooks
    assert (root / "04b_try_sam2_stamped_parts.ipynb") in notebooks
    assert (root / "04c_compare_pretrained_segmentation.ipynb") in notebooks
    assert (root / "04d_try_sam3_pretrained.ipynb") in notebooks
    assert (root / "04e_export_segmentation_review.ipynb") in notebooks
    for notebook in notebooks:
        data = json.loads(notebook.read_text(encoding="utf-8"))
        assert data["nbformat"] == 4 and data["metadata"]["kernelspec"]["name"] == "python3"
        assert len({cell["id"] for cell in data["cells"]}) == len(data["cells"])
        for cell in data["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None and not cell["outputs"]
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == "SOURCES" for t in node.targets):
                    sources = ast.literal_eval(node.value)
                    for name,source in sources.items():
                        ast.parse(source,filename=name)
                        if source and name not in ("roboqc/__init__.py","roboqc/intent/__init__.py"):
                            # Keep the old export helper in the user's untouched bootstrap cell.
                            if notebook.name != '04e_export_segmentation_review.ipynb' or name != 'roboqc/industrial/review_bundle.py':
                                assert source == (root.parent / "phase_1/src" / name).read_text(encoding="utf-8"), f"Rebuild embedded runtime: {name}"


def test_sam_notebook_accepts_private_photos_without_bundling_them():
    root = Path(__file__).resolve().parents[2] / "colab"
    cells = json.loads((root / "04b_try_sam2_stamped_parts.ipynb").read_text(encoding="utf-8"))["cells"]
    code_cells = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    source = next(s for s in code_cells if "paths = select_image_paths(IMAGE_PATH,RUN)" in s)
    assert (root / 'image_inputs.py').read_text(encoding='utf-8') in code_cells
    assert 'EMBEDDED_IMAGES' not in ''.join(code_cells)
    assert "download_demo_images" not in source and "urlopen" not in source
    for source in code_cells:
        calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)]
        assert not any(isinstance(n.func,ast.Attribute) and n.func.attr == "train" for n in calls)
    assert any("facebook/sam2.1-hiera-large" in s for s in code_cells)
    assert any("SAM2AutomaticMaskGenerator" in s for s in code_cells)
    assert any("predictor.predict(box=box" in s for s in code_cells)


def test_comparison_accepts_private_data_and_configures_five_distinct_methods():
    root = Path(__file__).resolve().parents[2] / "colab"
    cells = json.loads((root / "04c_compare_pretrained_segmentation.ipynb").read_text(encoding="utf-8"))["cells"]
    assignments = {}
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    for source in sources:
        for node in ast.parse(source).body:
            if isinstance(node,ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0],ast.Name):
                if node.targets[0].id == "METHODS":
                    assignments[node.targets[0].id] = ast.literal_eval(node.value)
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr == "train" for n in ast.walk(ast.parse(source)))
    assert assignments["METHODS"] == ["yoloe_text","sam2_auto","grounding_dino_sam2","owlv2_sam2","florence2_sam2"]
    assert 'EMBEDDED_IMAGES' not in ''.join(sources)
    assert (root / 'image_inputs.py').read_text(encoding='utf-8') in sources
    assert any('paths = select_image_paths(IMAGE_PATH,RUN)' in s for s in sources)
    assert any('expected = None' in s for s in sources)
    assert any("accuracy_winner" in s and "None" in s for s in sources)
    assert any("fewer than five methods completed" in s for s in sources)
    assert any("SAM2_CACHE = '/content/roboqc_sam2_cache'" in s for s in sources)
    assert any("MODEL_CACHE = '/content/roboqc_pretrained_cache'" in s for s in sources)
    assert any("CACHE = Path(MODEL_CACHE).resolve()" in s for s in sources)
    assert not any("pretrained_cache/comparison" in s for s in sources)
    assert any("'sam2_cache':SAM2_CACHE" in s and "traceback.print_exc(file=sys.stdout)" in s for s in sources)


def test_sam3_document_uses_cpu_cuda_local_cache_and_private_inputs():
    root = Path(__file__).resolve().parents[2] / "colab"
    cells = json.loads((root / "04d_try_sam3_pretrained.ipynb").read_text(encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    assert 'EMBEDDED_IMAGES' not in ''.join(sources)
    assert (root / 'image_inputs.py').read_text(encoding='utf-8') in sources
    assert any('paths = select_image_paths(IMAGE_PATH,RUN)' in s for s in sources)
    assert any("'facebook/sam3'" in s and "load_predictor(" in s for s in sources)
    assert not any("get_device_capability" in s or "SAM3_COMMIT" in s or "git+" in s for s in sources)
    assert any("DEVICE_CHOICE = 'auto'" in s for s in sources)
    assert any("MODEL_CACHE = '/content/roboqc_sam3_cache'" in s for s in sources)
    assert any("'sam3_' + uuid.uuid4().hex" in s for s in sources)
    assert any("if GPU: torch.cuda.synchronize()" in s for s in sources)
    assert any("torch.cuda.get_device_name(0) if GPU else None" in s for s in sources)
    assert any("result['kept_candidates']" in s for s in sources)
    assert any("userdata.get('HF_TOKEN')" in s and "del token" in s for s in sources)
    for source in sources:
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr in ["train","fit","from_pretrained"] for n in ast.walk(ast.parse(source)))


def test_reserved_grammars_preserve_quality_and_color_relations():
    for families in FAMILIES.values():
        for family in families:
            for description,mode,cat in [("buenas","quality","good"),("rojas","color","red"),("con defecto tipo 2","quality","defect_2")]:
                text = family.format(v="Pon",s="las piezas " + description,d="verde")
                response = understand(text)
                assert response["status"] == "valid", (text,response)
                assert response["command"] == {"intent":"sort","mode":mode,"rules":[{"category":cat,"destination":"green"}]}


def test_pretrained_route_never_trains_and_guards_labelled_evaluation():
    root = Path(__file__).resolve().parents[2] / "colab"
    quickstart = json.loads((root / "04a_try_yoloe_pretrained.ipynb").read_text(encoding="utf-8"))
    for cell in quickstart["cells"]:
        if cell["cell_type"] == "code":
            calls = [n for n in ast.walk(ast.parse("".join(cell["source"]))) if isinstance(n,ast.Call)]
            assert not any(isinstance(n.func,ast.Attribute) and n.func.attr == "train" for n in calls)
    evaluation = json.loads((root / "06_evaluate_vision.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in evaluation["cells"] if c["cell_type"] == "code"]
    assert any("EVALUATION_MODE = 'unlabelled'" in s for s in sources)
    labelled_cells = [s for s in sources if "source.rglob('instances.json')" in s or "truth_types, pred_types" in s]
    assert len(labelled_cells) == 2
    for source in labelled_cells:
        tree = ast.parse(source)
        assert len(tree.body) == 1 and isinstance(tree.body[0],ast.If)
        assert ast.unparse(tree.body[0].test) == "EVALUATION_MODE == 'labelled'"


def test_pretrained_and_evaluation_notebooks_offer_default_demo_photos():
    root = Path(__file__).resolve().parents[2] / "colab"
    for name in ["04a_try_yoloe_pretrained.ipynb","06_evaluate_vision.ipynb"]:
        cells = json.loads((root / name).read_text(encoding="utf-8"))["cells"]
        sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
        assert any("USE_DEMO_IMAGES = True" in s for s in sources)
        assert any("else download_demo_images(Path(ROOT) / 'demo_images') if USE_DEMO_IMAGES else upload_folder()" in s for s in sources)


def test_review_notebook_combines_six_methods_exports_drive_and_resumes_without_training():
    root = Path(__file__).resolve().parents[2] / "colab"
    notebook = json.loads((root / '04e_export_segmentation_review.ipynb').read_text(encoding='utf-8'))
    sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    config = next(s for s in sources if 'METHODS =' in s)
    methods = next(n.value for n in ast.parse(config).body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id == 'METHODS' for t in n.targets))
    assert ast.literal_eval(methods) == ['yoloe_text','sam2_auto','grounding_dino_sam2','owlv2_sam2','florence2_sam2','sam3_text']
    assert 'MAX_IMAGES = None' in config
    loading = next(s for s in sources if 'register_images(source,OUT)' in s)
    assert "source = Path(ROOT) / 'inputs'" in loading and 'extract_images(' in loading
    assert "sample['image_id']" in ''.join(sources)
    inference = next(s for s in sources if "index_path = OUT / 'proposals.json'" in s)
    assert 'load_predictor(' in inference and 'predict_image(' in inference and 'PretrainedMethod(' in inference
    assert "if method == 'sam3_text' else [(0,None)]" in inference
    assert 'finally:' in inference and 'gc.collect()' in inference and 'torch.cuda.empty_cache()' in inference
    assert "write_json(index_path,records)" in inference and "if load_error: raise RuntimeError" in inference
    assert "if GPU: torch.cuda.synchronize()" in inference
    export = next(s for s in sources if 'def export_saved_bundle(' in s)
    # The publication copy preserves the user's customized export implementation.
    # Check the export-only contract, not equality to a different helper revision.
    assert any(isinstance(n, ast.FunctionDef) and n.name == 'export_saved_bundle'
               for n in ast.parse(export).body)
    assert "EXPORT_FOLDER = Path(globals().get('ROOT'" in export
    assert "globals().get('EXPORT_LOCAL_DIR','/content/roboqc_review_exports')" in export
    assert not any(isinstance(n,ast.Name) and n.id=='EXPORT_LOCAL_DIR' for n in ast.walk(ast.parse(export)))
    assert 'Most export progress:' in export and 'Review checkpoints' in export and 'Review ZIP attempts' in export
    active_export_cell = next(c for c in notebook['cells'] if 'def export_saved_bundle(' in ''.join(c['source']))
    assert active_export_cell['execution_count'] is None and active_export_cell['outputs'] == []
    # Full rebuilds still provide optional auto inference resume. Do NOT rebuild this user-edited notebook.
    builder = (root / 'build_review_notebook.py').read_text(encoding='utf-8')
    assert "RESUME_BUNDLE = 'auto'" in builder and 'EXPORT_ONLY = False' in builder
    assert 'discover_bundles(ROOT,settings,' in builder and 'settings_compatible(previous,settings)' in builder
    assert 'EMBEDDED_IMAGES' not in ''.join(sources)
    for source in sources:
        calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)]
        assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ['fit','train'] for n in calls)


def test_pretrained_retries_use_new_bundle_and_preview_directories():
    root = Path(__file__).resolve().parents[2] / "colab"
    cells = json.loads((root / "04a_try_yoloe_pretrained.ipynb").read_text(encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    export = next(s for s in sources if "manifest = prepare_bundle(" in s)
    preview = next(s for s in sources if "report,rows = inspect_images(" in s)
    assert "'yoloe_bundle_' + uuid.uuid4().hex" in export
    assert "'yoloe_preview_' + uuid.uuid4().hex" in preview
    assert "inspect_images(OUT,images,PREVIEW," in preview
    assert "cv2.imread(str(PREVIEW / row['overlay']))" in preview


def test_voice_setup_does_not_combine_optimum_with_transformers_five():
    root = Path(__file__).resolve().parents[2] / "colab"
    for name in ["02_train_intent_model.ipynb","03_evaluate_intent_model.ipynb"]:
        cells = json.loads((root / name).read_text(encoding="utf-8"))["cells"]
        setup = next("".join(c["source"]) for c in cells if "import os, random, subprocess, sys, uuid" in "".join(c["source"]))
        assert "transformers>=5.2,<6" in setup
        assert "optimum" not in setup
        assert "stderr=subprocess.STDOUT" in setup and "roboqc_pip_install.log" in setup


def test_native_onnx_encoder_batches_tokens_without_optimum(tmp_path, monkeypatch):
    import sys
    import types
    import numpy as np
    from roboqc.industrial.models import ONNXSentenceEncoder
    (tmp_path / "onnx_config.json").write_text(json.dumps({"model_file":"model_int8.onnx","max_seq_length":8,"embedding_dim":1}))
    batches = []
    class Tokenizer:
        def __call__(self,texts,**kwargs):
            assert kwargs["max_length"] == 8 and kwargs["return_tensors"] == "np"
            return {"input_ids":np.array([[len(t),0] for t in texts]),"attention_mask":np.array([[1,0]]*len(texts))}
    class Session:
        def __init__(self,path,providers):
            assert Path(path).name == "model_int8.onnx" and providers == ["CPUExecutionProvider"]
        def get_inputs(self):
            return [types.SimpleNamespace(name=n) for n in ("input_ids","attention_mask")]
        def run(self,names,inputs):
            assert names == ["sentence_embedding"]
            assert all(v.dtype == np.int64 for v in inputs.values())
            batches.append(len(inputs["input_ids"]))
            return [inputs["input_ids"][:,:1].astype(np.float32)]
    monkeypatch.setitem(sys.modules,"onnxruntime",types.SimpleNamespace(InferenceSession=Session))
    monkeypatch.setitem(sys.modules,"transformers",types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path,local_files_only:Tokenizer())))
    encoder = ONNXSentenceEncoder(tmp_path)
    np.testing.assert_array_equal(encoder.encode(["a","abc","xy"],batch_size=2),[[1],[3],[2]])
    assert batches == [2,1] and encoder.encode([]).shape == (0,1)
