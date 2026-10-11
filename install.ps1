# GhostSwitch installer for Windows.
#
#   irm https://raw.githubusercontent.com/hiddenway/ghostswitch/main/install.ps1 | iex
#
# Everything goes into %LOCALAPPDATA%\GhostSwitch: uv, a private Python, the app, its libraries and,
# later, the models the app downloads. No administrator rights are needed and no system Python is
# used or changed. Running the same command again updates GhostSwitch.
#
# Options, as environment variables set before running ($env:GHOSTSWITCH_NO_START = "1"):
#   GHOSTSWITCH_ACTION=uninstall       remove GhostSwitch (asks before deleting downloaded models)
#   GHOSTSWITCH_NO_START=1             do not start GhostSwitch at the end
#   GHOSTSWITCH_NO_MODIFY_PATH=1       do not add the ghostswitch command to PATH
#   GHOSTSWITCH_VARIANT=cpu|cuda       the build to install instead of the detected one
#   GHOSTSWITCH_VERSION=v6.0.0         a release to install instead of the latest
#   GHOSTSWITCH_REF=main               a branch, tag or commit to install instead of a release
#   GHOSTSWITCH_SOURCE=path-or-url     a folder (from a git checkout, only its project files),
#                                      a .zip archive, or an archive URL
#   GHOSTSWITCH_ROOT=dir               where to install
# A variant, ref or source chosen this way is kept for updates; GHOSTSWITCH_VARIANT=auto and
# GHOSTSWITCH_REF=release or GHOSTSWITCH_SOURCE=release go back to the defaults.
#
# GhostSwitch was called OmniBrain. OMNIBRAIN_* variables still work, and a copy installed in
# %LOCALAPPDATA%\OmniBrain is updated where it is, with its models; its omnibrain command keeps
# working.
#
# The script runs inside the caller's PowerShell session (irm | iex), so it never calls exit,
# which would close that window; errors are thrown instead.

function Install-GhostSwitch {
    $ErrorActionPreference = "Stop"
    # Invoke-WebRequest is many times slower with its progress bar in Windows PowerShell 5.1.
    $ProgressPreference = "SilentlyContinue"
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    # An OMNIBRAIN_* variable from before the rename stands in for an unset GHOSTSWITCH_* one.
    foreach ($name in "ACTION", "ROOT", "REPO", "INSTALLER_URL", "SOURCE", "REF", "VARIANT", "VERSION",
                      "NO_START", "NO_MODIFY_PATH", "PURGE") {
        $legacy = [Environment]::GetEnvironmentVariable("OMNIBRAIN_$name")
        if ($legacy -and -not [Environment]::GetEnvironmentVariable("GHOSTSWITCH_$name")) {
            Set-Item "env:GHOSTSWITCH_$name" $legacy
        }
    }

    $UvVersion = "0.12.18"
    $PythonVersion = "3.12"
    $Root = $env:GHOSTSWITCH_ROOT
    if (-not $Root) {
        $Root = Join-Path $env:LOCALAPPDATA "GhostSwitch"
        $legacyRoot = Join-Path $env:LOCALAPPDATA "OmniBrain"
        if (-not (Test-Path $Root) -and (Test-Path (Join-Path $legacyRoot "bin\omnibrain.cmd"))) { $Root = $legacyRoot }
    }
    $Bin = Join-Path $Root "bin"
    $Programs = [Environment]::GetFolderPath("Programs")
    $Shortcut = Join-Path $Programs "GhostSwitch.lnk"
    $LegacyShortcut = Join-Path $Programs "OmniBrain.lnk"
    $ConfFile = Join-Path $Root "install.conf"

    function Fail([string]$Message) {
        throw "GhostSwitch could not be installed: $Message"
    }

    function Get-SavedSetting([string]$Name) {
        if (Test-Path $ConfFile) {
            foreach ($line in Get-Content $ConfFile) {
                foreach ($key in "GHOSTSWITCH_$Name", "OMNIBRAIN_$Name") {
                    if ($line -like "$key=*") { return $line.Substring($key.Length + 1) }
                }
            }
        }
        return ""
    }

    $Repo = $env:GHOSTSWITCH_REPO
    if (-not $Repo) { $Repo = Get-SavedSetting "REPO" }
    if (-not $Repo) { $Repo = "hiddenway/ghostswitch" }
    $DefaultInstallerUrl = "https://raw.githubusercontent.com/$Repo/main/install.ps1"
    $InstallerUrl = $env:GHOSTSWITCH_INSTALLER_URL
    if (-not $InstallerUrl) { $InstallerUrl = Get-SavedSetting "INSTALLER_URL" }
    if (-not $InstallerUrl) { $InstallerUrl = $DefaultInstallerUrl }
    # Where the app comes from: a local folder or archive, a git ref, or, by default, the latest release.
    $Source = $env:GHOSTSWITCH_SOURCE
    if (-not $Source) { $Source = Get-SavedSetting "SOURCE" }
    if ($Source -eq "release") { $Source = "" }
    if ($Source -and $Source -notmatch "^https?://" -and (Test-Path $Source)) { $Source = (Resolve-Path $Source).Path }
    $Ref = $env:GHOSTSWITCH_REF
    if (-not $Ref) { $Ref = Get-SavedSetting "REF" }
    if ($Ref -eq "release") { $Ref = "" }

    function Download([string]$Url, [string]$Path) {
        try { Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing }
        catch { Fail "could not download $Url ($($_.Exception.Message))" }
    }

    function Assert-Sha256([string]$Path, [string]$Expected) {
        $actual = (Get-FileHash -Algorithm SHA256 -Path $Path).Hash
        if (-not $Expected -or $actual -ne $Expected.Trim()) {
            Fail "$(Split-Path $Path -Leaf) is damaged (checksum mismatch); run the installer again."
        }
    }

    # Native programs report problems through exit codes and stderr; stderr text is not an error.
    function Invoke-Native([string]$File, [string[]]$Arguments) {
        $previous = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try { & $File @Arguments | Out-Host } finally { $ErrorActionPreference = $previous }
        return $LASTEXITCODE
    }

    function Stop-Running {
        # Updating or removing the app folder under a running server would fail on Windows.
        $processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase) }
        if ($processes) {
            $processes | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Write-Host "  Stopped the running GhostSwitch"
            Start-Sleep -Seconds 2
        }
    }

    function Get-UserPath {
        $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey("Environment")
        try { return [string]$key.GetValue("Path", "", [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames) }
        finally { $key.Close() }
    }

    function Set-UserPath([string]$Value) {
        # Written as an expandable string, so other entries keep their %VARIABLES%.
        $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey("Environment", $true)
        try { $key.SetValue("Path", $Value, [Microsoft.Win32.RegistryValueKind]::ExpandString) }
        finally { $key.Close() }
        # Setting any user variable this way tells Windows to reload the environment for new windows.
        [Environment]::SetEnvironmentVariable("GHOSTSWITCH_INSTALLER", "1", "User")
        [Environment]::SetEnvironmentVariable("GHOSTSWITCH_INSTALLER", $null, "User")
    }

    # --- Uninstall -----------------------------------------------------------------------------

    if ($env:GHOSTSWITCH_ACTION -eq "uninstall") {
        if (-not (Test-Path $Root)) {
            Write-Host "GhostSwitch is not installed in $Root."
            return
        }
        Stop-Running
        $keepData = $true
        $data = Join-Path $Root "data"
        if (Test-Path $data) {
            if ($env:GHOSTSWITCH_PURGE -eq "1") {
                $keepData = $false
            } elseif ([Environment]::UserInteractive) {
                $bytes = (Get-ChildItem $data -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
                $answer = Read-Host ("Also delete downloaded models and settings ({0:N1} GB in {1})? [y/N]" -f ($bytes / 1GB), $data)
                if ($answer -match "^(y|yes)$") { $keepData = $false }
            }
        }
        Get-ChildItem -Force $Root | Where-Object { -not ($keepData -and $_.Name -eq "data") } |
            Remove-Item -Recurse -Force
        if (-not (Get-ChildItem -Force $Root)) { Remove-Item -Force $Root }
        $parts = (Get-UserPath) -split ";" | Where-Object { $_ -and ($_.TrimEnd("\") -ine $Bin) }
        Set-UserPath ($parts -join ";")
        Remove-Item -Force $Shortcut, $LegacyShortcut -ErrorAction SilentlyContinue
        Write-Host "GhostSwitch is uninstalled."
        if ($keepData -and (Test-Path $data)) {
            Write-Host "Downloaded models and settings are still in $data; delete that folder to remove them."
        }
        return
    }

    # --- This computer -------------------------------------------------------------------------

    $arch = $env:PROCESSOR_ARCHITEW6432
    if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
    if ($arch -ne "AMD64") {
        Fail "GhostSwitch for Windows needs a 64-bit Intel or AMD processor; this computer reports $arch."
    }
    $detected = "cpu"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $report = (& nvidia-smi 2>$null) -join "`n"
        if ($report -match "CUDA Version:\s*(\d+)" -and [int]$Matches[1] -ge 12) { $detected = "cuda" }
    }
    # A build chosen once with GHOSTSWITCH_VARIANT stays chosen for updates; auto goes back to detection.
    $requested = $env:GHOSTSWITCH_VARIANT
    if (-not $requested) { $requested = Get-SavedSetting "VARIANT" }
    if (-not $requested -or $requested -eq "auto") {
        $requested = ""
        $variant = $detected
    } elseif ($requested -in @("cpu", "cuda")) {
        $variant = $requested
    } else {
        Fail "GHOSTSWITCH_VARIANT must be cpu, cuda or auto on Windows, not '$requested'."
    }
    if ($variant -eq "cuda") {
        Write-Host "Installing GhostSwitch for NVIDIA GPUs (llama.cpp with CUDA) into $Root"
    } else {
        Write-Host "Installing GhostSwitch (llama.cpp on the processor) into $Root"
    }

    New-Item -ItemType Directory -Force -Path $Bin | Out-Null
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("ghostswitch-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    try {
        # --- uv ---------------------------------------------------------------------------------
        $uv = Join-Path $Bin "uv.exe"
        $haveUv = $false
        if (Test-Path $uv) { $haveUv = ((& $uv --version) -split " ")[1] -eq $UvVersion }
        if (-not $haveUv) {
            Write-Host "  Downloading uv $UvVersion"
            $uvUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
            Download $uvUrl "$tmp\uv.zip"
            Download "$uvUrl.sha256" "$tmp\uv.zip.sha256"
            Assert-Sha256 "$tmp\uv.zip" ((Get-Content "$tmp\uv.zip.sha256" -Raw) -split "\s+")[0]
            Expand-Archive -Path "$tmp\uv.zip" -DestinationPath "$tmp\uv" -Force
            $found = Get-ChildItem "$tmp\uv" -Recurse -Filter uv.exe | Select-Object -First 1
            if (-not $found) { Fail "the uv archive has no uv.exe." }
            Copy-Item -Force $found.FullName $uv
        }

        # --- The app ----------------------------------------------------------------------------
        $src = Join-Path $tmp "src"
        New-Item -ItemType Directory -Force -Path $src | Out-Null
        if ($Source) {
            if ($Source -match "^https?://") {
                Write-Host "  Downloading the app from $Source"
                Download $Source "$tmp\app.zip"
                Expand-Archive -Path "$tmp\app.zip" -DestinationPath $src -Force
            } elseif (Test-Path $Source -PathType Container) {
                Write-Host "  Copying the app from $Source"
                $copy = Join-Path $src "app"
                $files = $null
                if (Get-Command git -ErrorAction SilentlyContinue) {
                    # A checkout: its tracked and new files, without what .gitignore leaves out (.venv, models).
                    $files = & git -C $Source -c core.quotepath=off ls-files --cached --others --exclude-standard 2>$null
                    if ($LASTEXITCODE -ne 0) { $files = $null }
                }
                if ($files) {
                    foreach ($file in $files) {
                        $from = Join-Path $Source $file
                        if (Test-Path -LiteralPath $from -PathType Leaf) {
                            $to = Join-Path $copy $file
                            New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
                            Copy-Item -LiteralPath $from -Destination $to
                        }
                    }
                } else {
                    Copy-Item -Recurse $Source $copy
                }
            } elseif (Test-Path $Source -PathType Leaf) {
                Write-Host "  Unpacking the app from $Source"
                if ($Source -match "\.(tar\.gz|tgz)$") {
                    tar.exe -xzf $Source -C $src
                    if ($LASTEXITCODE -ne 0) { Fail "could not unpack $Source." }
                } else {
                    Expand-Archive -Path $Source -DestinationPath $src -Force
                }
            } else {
                Fail "$Source does not exist."
            }
        } elseif ($Ref) {
            Write-Host "  Downloading GhostSwitch from github.com/$Repo at $Ref"
            Download "https://github.com/$Repo/archive/$Ref.zip" "$tmp\app.zip"
            Expand-Archive -Path "$tmp\app.zip" -DestinationPath $src -Force
        } else {
            if ($env:GHOSTSWITCH_VERSION) {
                $release = "https://github.com/$Repo/releases/download/$($env:GHOSTSWITCH_VERSION)"
                Write-Host "  Downloading GhostSwitch $($env:GHOSTSWITCH_VERSION)"
            } else {
                $release = "https://github.com/$Repo/releases/latest/download"
                Write-Host "  Downloading GhostSwitch (latest release)"
            }
            try {
                Download "$release/ghostswitch.zip" "$tmp\app.zip"
                Download "$release/SHA256SUMS" "$tmp\SHA256SUMS"
            } catch {
                Fail "could not download a release from github.com/$Repo/releases. To install the main branch instead, set `$env:GHOSTSWITCH_REF = 'main' and run the installer again."
            }
            $line = Get-Content "$tmp\SHA256SUMS" | Where-Object { $_ -match "\s\*?ghostswitch\.zip$" } | Select-Object -First 1
            Assert-Sha256 "$tmp\app.zip" (($line -split "\s+")[0])
            Expand-Archive -Path "$tmp\app.zip" -DestinationPath $src -Force
        }
        $project = Get-ChildItem $src -Recurse -Depth 1 -Filter pyproject.toml | Select-Object -First 1
        if (-not $project -or -not (Test-Path (Join-Path $project.DirectoryName "core\cli.py"))) {
            Fail "the app archive does not contain GhostSwitch."
        }
        $appNew = Join-Path $Root "app.new"
        if (Test-Path $appNew) { Remove-Item -Recurse -Force $appNew }
        Move-Item $project.DirectoryName $appNew

        # --- Python and libraries ---------------------------------------------------------------
        Stop-Running
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "python"
        $env:UV_CACHE_DIR = Join-Path $Root "cache"
        $env:UV_PROJECT_ENVIRONMENT = Join-Path $Root "venv"
        $env:UV_PYTHON_PREFERENCE = "only-managed"
        $env:UV_NO_CONFIG = "1"
        $python = Join-Path $Root "venv\Scripts\python.exe"

        function Sync-Library([string]$Extra) {
            if ($Extra -eq "cuda") {
                Write-Host "  Installing Python $PythonVersion and libraries (the NVIDIA build downloads about 2.5 GB)"
            } else {
                Write-Host "  Installing Python $PythonVersion and libraries"
            }
            $code = Invoke-Native $uv @("sync", "--project", $appNew, "--frozen", "--no-dev", "--extra", $Extra, "--python", $PythonVersion)
            if ($code -ne 0) { Fail "installing the libraries failed; check the internet connection and run the installer again." }
        }

        Sync-Library $variant
        if ($variant -eq "cuda") {
            $check = "from core import cuda_libraries; cuda_libraries.prepare(); import llama_cpp; raise SystemExit(0 if llama_cpp.llama_supports_gpu_offload() else 1)"
            Push-Location $appNew
            try { $code = Invoke-Native $python @("-c", $check) } finally { Pop-Location }
            if ($code -ne 0) {
                Write-Host "  The NVIDIA build does not load on this computer; installing the processor build instead."
                $variant = "cpu"
                Sync-Library "cpu"
            }
        }
        Push-Location $appNew
        try { $code = Invoke-Native $python @("-m", "core.cli", "--version") } finally { Pop-Location }
        if ($code -ne 0) { Fail "the installed app does not start; run the installer again." }
        Invoke-Native $uv @("cache", "prune", "--quiet") | Out-Null

        # --- Replace the app --------------------------------------------------------------------
        $app = Join-Path $Root "app"
        $appOld = Join-Path $Root "app.old"
        if (Test-Path $appOld) { Remove-Item -Recurse -Force $appOld }
        if (Test-Path $app) {
            # A window that was just closed can hold the folder for a moment.
            for ($attempt = 1; ; $attempt++) {
                try { Move-Item $app $appOld; break }
                catch {
                    if ($attempt -ge 10) { Fail "the app folder $app is in use; close GhostSwitch and run the installer again." }
                    Start-Sleep -Seconds 1
                }
            }
        }
        Move-Item $appNew $app
        if (Test-Path $appOld) { Remove-Item -Recurse -Force $appOld }
        $conf = @()
        if ($requested) { $conf += "GHOSTSWITCH_VARIANT=$requested" }
        if ($Source) { $conf += "GHOSTSWITCH_SOURCE=$Source" }
        if ($Ref) { $conf += "GHOSTSWITCH_REF=$Ref" }
        if ($Repo -ne "hiddenway/ghostswitch") { $conf += "GHOSTSWITCH_REPO=$Repo" }
        if ($InstallerUrl -ne $DefaultInstallerUrl) { $conf += "GHOSTSWITCH_INSTALLER_URL=$InstallerUrl" }
        if ($conf) { Set-Content -Path $ConfFile -Value $conf -Encoding ASCII } else { Remove-Item -Force $ConfFile -ErrorAction SilentlyContinue }
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }

    # --- The ghostswitch command -----------------------------------------------------------------
    # cmd.exe reads .cmd files in the console code page, so the launcher stays ASCII and finds its
    # folder at run time; a user name with other letters stays out of the file.
    $launcher = @(
        "@echo off",
        "rem Launcher written by the GhostSwitch installer.",
        "setlocal",
        "for %%I in (""%~dp0.."") do set ""GHOSTSWITCH_ROOT=%%~fI""",
        "set ""GHOSTSWITCH_HOME=%GHOSTSWITCH_ROOT%\data""",
        "set ""PYTHONUTF8=1""",
        "if /i ""%~1""==""update"" goto installer",
        "if /i ""%~1""==""uninstall"" goto installer",
        "rem The app comes from PYTHONPATH, not the current folder, which an update must be able to replace;",
        "rem -P keeps a 'core' folder wherever the user stands from shadowing the app.",
        "set ""PYTHONPATH=%GHOSTSWITCH_ROOT%\app""",
        """%GHOSTSWITCH_ROOT%\venv\Scripts\python.exe"" -P -m core.cli %*",
        "exit /b %ERRORLEVEL%",
        ":installer",
        "set ""GHOSTSWITCH_ACTION=%~1""",
        "rem One line, parsed before it runs: uninstall deletes this file while it runs.",
        "powershell -NoProfile -ExecutionPolicy Bypass -Command ""[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072; iex (irm '$InstallerUrl')"" && exit /b 0 || exit /b 1"
    )
    Set-Content -Path (Join-Path $Bin "ghostswitch.cmd") -Value $launcher -Encoding ASCII
    # A copy installed as OmniBrain keeps its omnibrain command.
    $legacyLauncher = Join-Path $Bin "omnibrain.cmd"
    if (Test-Path $legacyLauncher) { Set-Content -Path $legacyLauncher -Value $launcher -Encoding ASCII }

    if ($env:GHOSTSWITCH_NO_MODIFY_PATH -ne "1") {
        $parts = (Get-UserPath) -split ";" | Where-Object { $_ }
        if (-not ($parts | Where-Object { $_.TrimEnd("\") -ieq $Bin })) {
            Set-UserPath ((@($parts) + $Bin) -join ";")
        }
        try {
            $shell = New-Object -ComObject WScript.Shell
            $link = $shell.CreateShortcut($Shortcut)
            $link.TargetPath = Join-Path $Bin "ghostswitch.cmd"
            $link.WorkingDirectory = $Root
            $link.Description = "Start GhostSwitch"
            $link.Save()
            Remove-Item -Force $LegacyShortcut -ErrorAction SilentlyContinue
        } catch {
            Write-Host "  (Could not add GhostSwitch to the Start menu: $($_.Exception.Message))"
        }
    }
    if (-not (($env:Path -split ";") | Where-Object { $_.TrimEnd("\") -ieq $Bin })) {
        $env:Path = "$env:Path;$Bin"
    }

    Push-Location (Join-Path $Root "app")
    try { $version = (& $python -m core.cli --version) } finally { Pop-Location }
    Write-Host ""
    Write-Host "$version is installed."
    Write-Host "  Start it any time with:  ghostswitch   (or GhostSwitch in the Start menu)"
    Write-Host "  Update:     ghostswitch update"
    Write-Host "  Uninstall:  ghostswitch uninstall"
    Write-Host ""

    if ($env:GHOSTSWITCH_NO_START -ne "1") {
        & (Join-Path $Bin "ghostswitch.cmd")
    }
}

Install-GhostSwitch
