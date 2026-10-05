# CosyVoice3 服务端测试脚本（在客户端机器运行）
# 用法:
#   .\test_server.ps1                              # 默认 http://192.168.1.2:9880
#   .\test_server.ps1 -Base http://192.168.1.2:9880 -PromptWav D:\path\ref.wav
# 输出: out\*.wav，双击直接听

param(
    [string]$Base = "http://192.168.1.2:50000",
    [string]$PromptWav = (Join-Path $PSScriptRoot "asset\zero_shot_prompt.wav")
)

$ErrorActionPreference = "Continue"
$outdir = Join-Path $PWD "out"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

$script:pass = 0
$script:fail = 0
$PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

function Check([string]$Name, [string]$File, [int]$MinSize = 44) {
    if (Test-Path $File) {
        $len = (Get-Item $File).Length
        if ($len -ge $MinSize) {
            Write-Host "  PASS $Name -> $File ($len bytes)" -ForegroundColor Green
            $script:pass++
        } else {
            Write-Host "  FAIL $Name -> $File 只有 $len bytes，错误内容:" -ForegroundColor Red
            Get-Content $File -Raw -ErrorAction SilentlyContinue
            $script:fail++
        }
    } else {
        Write-Host "  FAIL $Name 没有输出文件" -ForegroundColor Red
        $script:fail++
    }
}

function Invoke-Post([string]$Url, [string]$Out, [string[]]$Form) {
    $curlArgs = @("-s", "-X", "POST", $Url, "--max-time", "600")
    foreach ($f in $Form) { $curlArgs += @("-F", $f) }
    $curlArgs += @("-o", $Out, "-w", "[http_code=%{http_code} time=%{time_total}s]")
    $r = & curl.exe @curlArgs 2>&1
    Write-Host "  $r"
}

Write-Host "Base: $Base"

# 1. 连通性检查（/v1/models）
Write-Host "`n=== 1. 连通性 ==="
try {
    $h = Invoke-RestMethod -Uri "$Base/v1/models" -TimeoutSec 15
    Write-Host ("  models=[{0}]" -f (($h.data | ForEach-Object { $_.id }) -join ","))
    $script:pass++
} catch {
    Write-Host "  FAIL $Base/v1/models 连不上: $_" -ForegroundColor Red
    Write-Host "服务没起来或端口不对，用 -Base 指定。" -ForegroundColor Yellow
    exit 1
}

# 2. zero_shot 上传参考音频
Write-Host "`n=== 2. zero_shot（上传参考音频）==="
Invoke-Post "$Base/tts/zero_shot" "$outdir\zero_shot.wav" @(
    "text=你好，这是 CosyVoice3 测试。",
    "prompt_text=$PROMPT_TEXT",
    "prompt_wav=@$PromptWav"
)
Check "zero_shot" "$outdir\zero_shot.wav"

# 3. zero_shot 默认音色
Write-Host "`n=== 3. zero_shot（默认音色，不传 prompt_wav）==="
Invoke-Post "$Base/tts/zero_shot" "$outdir\zero_shot_default.wav" @(
    "text=你好，这是默认音色。",
    "prompt_text=$PROMPT_TEXT"
)
Check "zero_shot_default" "$outdir\zero_shot_default.wav"

# 4. cross_lingual
Write-Host "`n=== 4. cross_lingual ==="
Invoke-Post "$Base/tts/cross_lingual" "$outdir\cross_lingual.wav" @(
    "text=Hello, this is CosyVoice3."
)
Check "cross_lingual" "$outdir\cross_lingual.wav"

# 5. instruct2
Write-Host "`n=== 5. instruct2 ==="
Invoke-Post "$Base/tts/instruct2" "$outdir\instruct2.wav" @(
    "text=你好，这是 CosyVoice3 测试。",
    "instruct=You are a helpful assistant.<|endofprompt|>请用四川话说出：",
    "prompt_wav=@$PromptWav"
)
Check "instruct2" "$outdir\instruct2.wav"

# 6. sft（模型带 SFT 音色才测）
Write-Host "`n=== 6. sft ==="
if ($h.sft_voices -and $h.sft_voices.Count -gt 0) {
    $spk = $h.sft_voices[0]
    Write-Host "  spk_id=$spk"
    Invoke-Post "$Base/tts/sft" "$outdir\sft.wav" @(
        "text=你好，这是 SFT 音色。",
        "spk_id=$spk"
    )
    Check "sft" "$outdir\sft.wav"
} else {
    Write-Host "  无 SFT 音色，跳过"
}

# 7. 音色库 CRUD
Write-Host "`n=== 7. 音色库 ==="
$reg = @("-s", "-X", "POST", "$Base/voices", "-F", "name=tester", "-F", "file=@$PromptWav", "-w", "[http_code=%{http_code}]")
Write-Host "  register: $(& curl.exe @reg 2>&1)"
Invoke-Post "$Base/tts/zero_shot" "$outdir\voice_tester.wav" @(
    "text=你好，这是注册音色。",
    "prompt_text=$PROMPT_TEXT",
    "voice=tester"
)
Check "voice=tester" "$outdir\voice_tester.wav"
$del = @("-s", "-X", "DELETE", "$Base/voices/tester", "-w", "[http_code=%{http_code}]")
Write-Host "  delete:   $(& curl.exe @del 2>&1)"

# 8. stream=true（注意：输出是多个 wav 首尾拼接，不是单个合法 wav）
Write-Host "`n=== 8. stream=true ==="
Invoke-Post "$Base/tts/zero_shot" "$outdir\stream.wav" @(
    "text=你好，这是流式输出测试。第一句结束。第二句开始。",
    "prompt_text=$PROMPT_TEXT",
    "prompt_wav=@$PromptWav",
    "stream=true"
)
if (Test-Path "$outdir\stream.wav") {
    Write-Host ("  stream -> {0} bytes（多个 wav 拼接，仅供参考）" -f (Get-Item "$outdir\stream.wav").Length)
}

# 9. OpenAI 兼容
Write-Host "`n=== 9. OpenAI 兼容 ==="
$models = & curl.exe -s "$Base/v1/models" 2>&1
Write-Host "  /v1/models: $models"
# 中文 JSON 不能走命令行传参（控制台代码页会破坏编码），写文件后用 -d @file
$json = '{"model":"cosyvoice-zero-shot","input":"你好，这是 OpenAI 兼容接口测试。","voice":"default"}'
$reqFile = Join-Path $outdir "openai_req.json"
[IO.File]::WriteAllText($reqFile, $json, (New-Object System.Text.UTF8Encoding($false)))
$oa = @("-s", "-X", "POST", "$Base/v1/audio/speech", "-H", "Content-Type: application/json", "-d", "@$reqFile", "-o", "$outdir\openai.wav", "-w", "[http_code=%{http_code} time=%{time_total}s]")
Write-Host "  $(& curl.exe @oa 2>&1)"
Check "openai" "$outdir\openai.wav"

Write-Host "`n=============================="
Write-Host ("完成: {0} 通过, {1} 失败。wav 在 out\ 目录。" -f $script:pass, $script:fail)
