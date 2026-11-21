"""Prediction API for Model B, CRNN-CTC (Contract 5). STUB: Nikhil replaces this."""

# ===== CONCEPTS USED IN THIS FILE =====
# This file is only a STUB: it has no imports and no model code. Both functions return fixed values.
# Contract 5: the project's agreed prediction interface, one predict.py per model (see CONTRACTS.md).
#    load_model(ckpt_path, device) loads the model, and predict(model, image) returns the recognised text.
# Stub: a placeholder with the final function names and arguments, so code that calls the API can run
#    before the real model is ready (CONTRACTS.md: "so the demo works from week 1").
# Checkpoint path and device: the arguments of load_model. The stub accepts them but does not use them.

# ===== VIVA Q&A FOR THIS FILE =====
# Q: What part of the project is this file, and is it finished?
# A: It is the prediction entry point of the CRNN-CTC model (Contract 5). It is NOT finished: it is a stub
#    that returns None and a fixed text. The real loading and decoding code is still to be written.
# Q: Why does a stub with the final function names exist?
# A: Contract 5 fixes the names and arguments, so other code can call load_model and predict now and keep
#    working unchanged when the real model replaces the stub. The text "(model not ready yet)" shows it is a stub.


def load_model(ckpt_path: str, device: str = "cpu"):
    # Q: What does it do and why does it exist?
    # A: Now it only returns None and ignores both arguments. By Contract 5 it should load the trained model
    #    and what it needs (vocab) from ckpt_path. The stub keeps the function name and signature in place.
    # Q: What bug would appear if the stub were used as if it were the real function?
    # A: The returned model is None, so nothing real is loaded. A caller that uses the model (not only
    #    passes it to predict) would fail with an error on None, or silently show the wrong output.
    return None


def predict(model, image) -> str:
    # Q: What does it do and why does it exist?
    # A: Now it returns the fixed text "(model not ready yet)" for any input and does not look at the model
    #    or the image. By Contract 5 it should return the recognised word. It lets a demo run before the model.
    # Q: What bug would appear if this stub is left in by mistake?
    # A: Every image would get the same text, with no error, so a result could look fine but be meaningless.
    #    The visible placeholder text makes that mistake easy to notice.
    return "(model not ready yet)"
