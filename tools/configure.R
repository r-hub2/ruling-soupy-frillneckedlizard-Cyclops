# Gets run when compiling

makevars_in <- file.path("src", "Makevars.in")
makevars_win_in <- file.path("src", "Makevars.win.in")

makevars_out <- file.path("src", "Makevars")
makevars_win_out <- file.path("src", "Makevars.win")

txt <- readLines(makevars_in)
txt_win <- readLines(makevars_win_in)
if (getRversion() < "4.3") { # macOS / linux
    if (!any(grepl("^CXX_STD", txt))) {
        txt <- c("CXX_STD = CXX11", txt)
    }
}

if (getRversion() < "4.2") { # Windoz
    if (!any(grepl("^CXX_STD", txt_win))) {
        txt_win <- c("CXX_STD = CXX11", txt_win)
    }
}

##### can find JNI #####

doJni <- FALSE
if (doJni) {

if (.Platform$OS.type == "unix") {
    java_home <- Sys.getenv("JAVA_HOME")
} else {
    system("make -C tools/jvm-w32")
    java_home <- system("tools/jvm-w32/findjava -s -f", intern = TRUE)
    system("make -C tools/jvm-w32 clean")
}

if (nchar(java_home) == 0) {
    message("No JAVA_HOME defined; ignoring JNI compilation")
} else {
    message("Using JAVA_HOME=", java_home)
    jni_path = file.path(java_home, "include")

    # modify Makevars.in if Java is available
    txt[grep("^PKG_CPPFLAGS", txt)] <- paste(txt[grep("^PKG_CPPFLAGS", txt)],
                                             paste0("-I", jni_path),
                                             paste0("-I", jni_path, "/darwin"), # TODO Make OS-dependent
                                             paste0("-I", jni_path, "/linux"))
    engine_idx <- grep("^OBJECTS.engine", txt)
    txt[engine_idx+1] <- paste(txt[engine_idx+1],
                               "cyclops/jni/dr_inference_regression_RegressionJNIWrapper.o",
                               "cyclops/jni/dr_inference_regression_NewRegressionJNIWrapper.o")
    # Handle Windows
    txt_win[grep("^PKG_CPPFLAGS", txt_win)] <- paste(txt_win[grep("^PKG_CPPFLAGS", txt_win)],
                                             paste0("-I", jni_path),
                                             paste0("-I", jni_path, "/win32"))

    sources_idx <- grep("^SOURCES", txt_win)
    txt_win[sources_idx + 1] <- paste0(txt_win[sources_idx + 1],
                                      "\n\t\t\t\t\tcyclops/jni/*.cpp \\")
}

} # doJNI

#################### CUDA Toolkit ####################

cuda_home <- system2(command = "find", args = c("/usr/local/", "-maxdepth", "1" ,"-name", "cuda"), stdout  = TRUE)
if (TRUE || nchar(cuda_home)==0) { # By default, no CUDA build
    message("no CUDA installation found; only compile host code")
} else {
    message(paste0("using CUDA_HOME=", cuda_home))
    nvcc <- c(paste0(cuda_home, "/bin/nvcc -arch=sm_70")) # TODO remove hardcoding of ARCH
    cub_path <- paste0(cuda_home ,"/include") # CUB is included since CUDA Toolkit 11.0

    # whether this is the 64 bit linux version of CUDA
    cu_libdir <- system2(command = "find", args = c(paste0(cuda_home ,"/lib64")), stdout  = TRUE)
    if (nchar(cu_libdir) == 0) {
        cu_libdir <- paste0(cuda_home ,"/lib")
    }

    cuda_libs <- paste0("-L", cu_libdir, " -lcudart")
    cuda_cppflags <-paste0("-DHAVE_CUDA -I", cuda_home, "/include -I", cu_libdir, " -pthread -rdynamic")

    # modify Makevars.in if CUDA is available
    txt[grep("^PKG_LIBS", txt)] <- paste(txt[grep("^PKG_LIBS", txt)], cuda_libs)
    txt[grep("^PKG_CPPFLAGS", txt)] <- paste(txt[grep("^PKG_CPPFLAGS", txt)], cuda_cppflags)
    engine_idx <- grep("^OBJECTS.engine", txt)
    txt[engine_idx+1] <- paste(txt[engine_idx+1],
                               "cyclops/engine/CudaKernel.o",
                               "cyclops/engine/CudaDetail.o")
    txt <- c(txt,
             'all: $(OBJECTS)',
             '%.o: %.cu',
             paste0('\t', nvcc,' --default-stream per-thread -c -Xcompiler "-fPIC $(CPPFLAGS) -c" -I', cub_path,' $(R_PATH_LINKER) $^ -o $@'),
             'clean:',
             '\trm -rf *o',
             '.PHONY: all clean')
}

######################################################

if (.Platform$OS.type == "unix") {
	cat(txt, file = makevars_out, sep = "\n")
} else {
	cat(txt_win, file = makevars_win_out, sep = "\n")
}
