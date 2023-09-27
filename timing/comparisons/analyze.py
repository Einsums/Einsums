import matplotlib.pyplot as plot
import math
import os

def print_plots() :
    mkl_x = []
    mkl_y = []
    hip_copy_x = []
    hip_copy_y = []
    hip_map_x = []
    hip_map_y = []
    rocm_copy_x = []
    rocm_copy_y = []
    rocm_map_x = []
    rocm_map_y = []
    copy_large_x = []
    copy_large_y = []
    map_large_x = []
    map_large_y = []

    # Read the CSVs.
    with open("mkl_out.csv", "r") as fp :
        for line in fp :
            toks = line.strip().split(',')
            mkl_x.append(float(toks[0]))
            mkl_y.append(float(toks[1]))
            
    with open("hip_copy_out.csv", "r") as fp :
        for line in fp :
            toks = line.strip().split(',')
            hip_copy_x.append(float(toks[0]))
            hip_copy_y.append(float(toks[1]))

    with open("hip_map_out.csv", "r") as fp :
        for line in fp :
            toks = line.strip().split(',')
            hip_map_x.append(float(toks[0]))
            hip_map_y.append(float(toks[1]))

    with open("rocm_copy_out.csv", "r") as fp :
        for line in fp :
            toks = line.strip().split(',')
            rocm_copy_x.append(float(toks[0]))
            rocm_copy_y.append(float(toks[1]))

    with open("rocm_map_out.csv", "r") as fp :
        for line in fp :
            toks = line.strip().split(',')
            rocm_map_x.append(float(toks[0]))
            rocm_map_y.append(float(toks[1]))

    with open("hip_large_copy_out.csv", "r") as fp :
        for line in fp :
            toks = line.strip().split(',')
            copy_large_x.append(float(toks[0]))
            copy_large_y.append(float(toks[1]))
            
    with open("hip_large_map_out.csv", "r") as fp :
        for line in fp :
            toks = line.strip().split(',')
            map_large_x.append(float(toks[0]))
            map_large_y.append(float(toks[1]))
        

    # Plot the different figures.
    plot.figure(dpi=200)
    plot.scatter(mkl_x, mkl_y, label="MKL")
    plot.scatter(hip_map_x, hip_map_y, label="Mapped")
    plot.scatter(hip_copy_x, hip_copy_y, label="Copied")

    plot.legend()
    plot.xlabel("Matrix rows")
    plot.ylabel("Clock runtime")
    plot.title("NxN Matrix Multiplication")

    plot.savefig("fig1.png")

    plot.figure(dpi = 200)
    plot.scatter(hip_map_x, hip_map_y, label="HIP Mapped")
    plot.scatter(hip_copy_x, hip_copy_y, label="HIP Copied")
    plot.scatter(rocm_map_x, rocm_map_y, label="ROCm Mapped")
    plot.scatter(rocm_copy_x, rocm_copy_y, label="ROCm Copied")

    plot.legend()
    plot.xlabel("Matrix rows")
    plot.ylabel("Clock runtime")
    plot.title("NxN Matrix Multiplication: GPU Only")

    plot.savefig("fig2.png")

    plot.figure(dpi = 200)
    plot.scatter(hip_map_x, hip_map_y, label="HIP")
    plot.scatter(rocm_map_x, rocm_map_y, label="ROCm")

    plot.legend()
    plot.xlabel("Matrix rows")
    plot.ylabel("Clock runtime")
    plot.title("NxN Matrix Multiplication: Mapped Only")

    plot.savefig("fig3.png")

    plot.figure(dpi = 200)
    plot.scatter(map_large_x, map_large_y, label="Mapped")
    plot.scatter(copy_large_x, copy_large_y, label="Copied")

    plot.legend()
    plot.xlabel("Matrix rows")
    plot.ylabel("Clock runtime")
    plot.title("NxN Matrix Multiplication: GPU Comparison on Large Input")

    plot.savefig("fig4.png")

if __name__ == "__main__" :
    print_plots()
