import os
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from ctpamap import __version__

app = FastAPI(title="CTPAMAP API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict = {}
JOBS: dict = {}


class LoadRequest(BaseModel):
    pre: str
    ctpa: str
    out: Optional[str] = None


class RunRequest(BaseModel):
    session_id: str
    params: dict = {}


@app.get("/api/status")
def status():
    return {"ok": True, "version": __version__, "python": sys.version}


@app.post("/api/load")
def load(req: LoadRequest):
    import nibabel as nib

    session_id = str(uuid.uuid4())
    out_dir = req.out or str(Path(req.pre).parent / "ctpamap_output")

    pre_img = nib.load(req.pre)
    ctpa_img = nib.load(req.ctpa)
    meta = {
        "pre_shape": list(pre_img.shape),
        "ctpa_shape": list(ctpa_img.shape),
        "pre_spacing": [float(x) for x in pre_img.header.get_zooms()[:3]],
        "ctpa_spacing": [float(x) for x in ctpa_img.header.get_zooms()[:3]],
    }

    SESSIONS[session_id] = {
        "pre": req.pre,
        "ctpa": req.ctpa,
        "out_dir": out_dir,
        "meta": meta,
    }
    return {"session_id": session_id, "meta": meta}


def _run_background(job_id: str, session: dict, params: dict):
    from ctpamap.cli import run_pipeline

    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["progress_pct"] = 0

    def progress_cb(stage, pct):
        JOBS[job_id]["progress_pct"] = pct
        JOBS[job_id]["stage"] = stage

    try:
        run_pipeline(
            session["pre"],
            session["ctpa"],
            session["out_dir"],
            target_spacing=tuple(params.get("spacing", [1.5, 1.5, 1.5])),
            skip_reg=params.get("skip_reg", False),
            aggressive=params.get("aggressive", False),
            method=params.get("method", "default"),
            hd=params.get("hd", False),
            lite=params.get("lite", False),
            no_seg=params.get("no_seg", False),
            do_aer=params.get("do_aer", False),
            use_gpu=params.get("use_gpu", False),
            no_clip=params.get("no_clip", False),
            rigid_z_search=params.get("rigid_z_search", False),
            progress_callback=progress_cb,
        )
        out_dir = Path(session["out_dir"])
        outputs = {}
        for name in [
            "ctpa_regrid",
            "registered_pre",
            "lung_mask",
            "vessel_mask",
            "airway_mask",
            "iodine_map_raw",
            "iodine_map_clean",
            "iodine_overlay",
        ]:
            p = out_dir / f"{name}.nii.gz"
            if p.exists():
                outputs[name] = str(p)
        qc_path = out_dir / "qc_report.json"
        if qc_path.exists():
            outputs["qc_report"] = str(qc_path)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["progress_pct"] = 100
        JOBS[job_id]["outputs"] = outputs
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.post("/api/run")
def run(req: RunRequest, background_tasks: BackgroundTasks):
    if req.session_id not in SESSIONS:
        return JSONResponse(status_code=404, content={"error": "session not found"})
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "progress_pct": 0, "stage": None, "outputs": None, "error": None}
    background_tasks.add_task(
        _run_background, job_id, SESSIONS[req.session_id], req.params
    )
    return {"job_id": job_id}


@app.get("/api/run/{job_id}")
def get_run(job_id: str):
    if job_id not in JOBS:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return JOBS[job_id]


@app.get("/api/nifti")
def get_nifti(path: str = Query(...)):
    if not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"error": "file not found"})
    return FileResponse(path, media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=18923)
