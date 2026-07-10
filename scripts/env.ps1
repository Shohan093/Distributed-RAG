# Local-only dev convenience for Windows: points PySpark at a portable JDK
# 17 + winutils shim under .tools/ (see README "Local smoke test" section)
# so `local[*]` Spark works without a system-wide Java/Hadoop install. Not
# needed on the docker-compose/VM path (Linux containers ship their own JRE
# and don't need winutils at all -- see DEPLOYMENT.md).
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ToolsRoot = Join-Path $RepoRoot ".tools"
$env:JAVA_HOME = Join-Path $ToolsRoot "jdk-17.0.19+10"
$env:HADOOP_HOME = Join-Path $ToolsRoot "hadoop"
$env:PATH = "$env:HADOOP_HOME\bin;$env:JAVA_HOME\bin;$env:PATH"
Write-Output "JAVA_HOME=$env:JAVA_HOME"
Write-Output "HADOOP_HOME=$env:HADOOP_HOME"
