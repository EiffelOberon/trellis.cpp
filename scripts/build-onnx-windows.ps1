<# Build the native CUDA library, optional TensorRT SS runtime, and worker harness. #>
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-onnx',
    [string]$TensorRTRoot = 'out/TensorRT',
    [string]$VisualStudioRoot = '',
    [int]$Parallel = 6,
    [switch]$Test
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    if (-not $VisualStudioRoot) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        $VisualStudioRoot = & $vswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (-not $VisualStudioRoot) { throw 'Visual Studio 2022 C++ tools not found; supply -VisualStudioRoot' }
    }
    Import-Module (Join-Path $VisualStudioRoot 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $VisualStudioRoot -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
    $sdk = (Resolve-Path -LiteralPath $TensorRTRoot).Path
    if (-not (Test-Path -LiteralPath (Join-Path $sdk 'include\NvInfer.h'))) { throw 'TensorRT headers not found' }
    cmake -S . -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_OPENMP=OFF -DTRELLIS_TENSORRT=ON "-DTENSORRT_ROOT=$sdk"
    if ($LASTEXITCODE) { throw 'CMake configure failed' }
    cmake --build $BuildDir --target trellis-cli trellis-server trellis-job trellis-test-job trellis-test-trt trellis-compare-ss trellis-test-streamed-flow trellis-test-sparse-chunk --parallel $Parallel
    if ($LASTEXITCODE) { throw 'Native build failed' }
    if ($Test) {
        # CUDA 13 moved Windows runtime DLLs into bin/x64.
        $env:PATH = (Join-Path $env:CUDA_PATH 'bin\x64') + ';' + $env:PATH
        ctest --test-dir $BuildDir --output-on-failure
        if ($LASTEXITCODE) { throw 'Contract tests failed' }
    }
} finally { Pop-Location }
