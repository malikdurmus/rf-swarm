import numpy as np

xp = np 
IS_GPU_BACKEND = False


def set_backend(use_gpu : bool):
    """
    Sets the numerical backend for all calculations.

    This function should be called once at the start of the program.

    Args:
        use_gpu (bool): If True, attempts to use CuPy. Falls back to NumPy if
                        CuPy is not available. If False, uses NumPy.
    """
    global xp, IS_GPU_BACKEND

    if use_gpu:
        try:
            import cupy as cp
            xp = cp
            IS_GPU_BACKEND = True
            print("✅ Backend set to GPU (CuPy).")
        except ImportError:
            print("⚠️  Could not import CuPy. Falling back to CPU (NumPy) backend.")
            xp = np
            IS_GPU_BACKEND = False
    else: 
        xp = np
        IS_GPU_BACKEND = False
        print("✅ Backend set to CPU (NumPy).")


    
