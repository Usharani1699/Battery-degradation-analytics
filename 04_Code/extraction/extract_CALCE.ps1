# CALCE Battery Dataset — Extract tab-delimited .txt files to CSV
# Run via: powershell -ExecutionPolicy Bypass -File extract_CALCE.ps1

param()
$base   = "C:\Users\pure ev\Music\Term 3\CALCE Battery Dataset"
$outDir = "C:\Users\pure ev\Music\Term 3\Extracted_CSV_Data\CALCE"
$tmpDir = "C:\Users\pure ev\Music\Term 3\_calce_tmp"

foreach ($d in @($outDir, $tmpDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force $d | Out-Null }
}

$groups = @(
    [pscustomobject]@{ Name="CS2_LCO";  ZipFolder="CS"                   },
    [pscustomobject]@{ Name="CX2_LCO";  ZipFolder="CX"                   },
    [pscustomobject]@{ Name="A123_LFP"; ZipFolder="A123 Battery"          },
    [pscustomobject]@{ Name="INR_NMC";  ZipFolder="INR 18650-20R Battery" }
)

foreach ($grp in $groups) {
    $zipFolder = Join-Path $base $grp.ZipFolder
    if (-not (Test-Path $zipFolder)) { Write-Host "SKIP: $zipFolder not found"; continue }

    $zips = Get-ChildItem $zipFolder -Filter "*.zip" -Recurse
    if ($zips.Count -eq 0) { Write-Host "SKIP: no zips in $zipFolder"; continue }

    Write-Host "`nGroup: $($grp.Name) - $($zips.Count) zip files"
    $allRows = [System.Collections.Generic.List[object]]::new()

    foreach ($zip in $zips) {
        $batName  = [System.IO.Path]::GetFileNameWithoutExtension($zip.FullName)
        $unzipDir = Join-Path $tmpDir $batName
        if (Test-Path $unzipDir) { Remove-Item $unzipDir -Recurse -Force }
        try {
            Expand-Archive -Path $zip.FullName -DestinationPath $unzipDir -Force 2>$null
        } catch { Write-Host "  Unzip failed: $($zip.Name)"; continue }

        $txtFiles = Get-ChildItem $unzipDir -Filter "*.txt" -Recurse | Sort-Object Name
        $batRows  = 0

        foreach ($tf in $txtFiles) {
            $lines = [System.IO.File]::ReadAllLines($tf.FullName)
            if ($lines.Count -lt 2) { continue }

            # Parse header into index map
            $headers = $lines[0] -split "`t"
            $hi = @{}
            for ($c = 0; $c -lt $headers.Count; $c++) { $hi[$headers[$c].Trim()] = $c }

            # Get column indices (inline, no nested functions)
            $iTime   = if ($hi.ContainsKey("Time"))             { $hi["Time"] }             else { -1 }
            $iMV     = if ($hi.ContainsKey("mV"))               { $hi["mV"] }               else { -1 }
            $iMA     = if ($hi.ContainsKey("mA"))               { $hi["mA"] }               else { -1 }
            $iTemp   = if ($hi.ContainsKey("Temperature"))      { $hi["Temperature"] }      else { -1 }
            $iCap    = if ($hi.ContainsKey("Capacity"))         { $hi["Capacity"] }         else { -1 }
            $iChrg   = if ($hi.ContainsKey("Charge count"))     { $hi["Charge count"] }     else { -1 }
            $iDisc   = if ($hi.ContainsKey("Discharge count"))  { $hi["Discharge count"] }  else { -1 }
            $iPgm    = if ($hi.ContainsKey("Pgm cycle"))        { $hi["Pgm cycle"] }        else { -1 }
            $iStat   = if ($hi.ContainsKey("Status category"))  { $hi["Status category"] }  else { -1 }

            for ($r = 1; $r -lt $lines.Count; $r++) {
                $line = $lines[$r]
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                $p = $line -split "`t"

                $mv  = if ($iMV -ge 0 -and $iMV -lt $p.Count) { $p[$iMV].Trim() } else { "" }
                $ma  = if ($iMA -ge 0 -and $iMA -lt $p.Count) { $p[$iMA].Trim() } else { "" }
                $vv  = if ($mv -ne "") { try { [math]::Round([double]$mv / 1000.0, 5) } catch { $mv } } else { "" }
                $av  = if ($ma -ne "") { try { [math]::Round([double]$ma / 1000.0, 5) } catch { $ma } } else { "" }

                $allRows.Add([pscustomobject]@{
                    Battery_ID      = $batName
                    Group           = $grp.Name
                    File            = $tf.Name
                    Time_s          = if ($iTime -ge 0 -and $iTime -lt $p.Count) { $p[$iTime].Trim() } else { "" }
                    Pgm_Cycle       = if ($iPgm  -ge 0 -and $iPgm  -lt $p.Count) { $p[$iPgm].Trim()  } else { "" }
                    Charge_Count    = if ($iChrg -ge 0 -and $iChrg -lt $p.Count) { $p[$iChrg].Trim() } else { "" }
                    Discharge_Count = if ($iDisc -ge 0 -and $iDisc -lt $p.Count) { $p[$iDisc].Trim() } else { "" }
                    Status_Category = if ($iStat -ge 0 -and $iStat -lt $p.Count) { $p[$iStat].Trim() } else { "" }
                    Voltage_V       = $vv
                    Current_A       = $av
                    Temperature_C   = if ($iTemp -ge 0 -and $iTemp -lt $p.Count) { $p[$iTemp].Trim() } else { "" }
                    Capacity_raw    = if ($iCap  -ge 0 -and $iCap  -lt $p.Count) { $p[$iCap].Trim()  } else { "" }
                })
                $batRows++
            }
        }

        Write-Host "  $($batName): $batRows rows"
        Remove-Item $unzipDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    if ($allRows.Count -gt 0) {
        $csv = Join-Path $outDir "$($grp.Name)_AllCycles.csv"
        $allRows | Export-Csv -Path $csv -NoTypeInformation -Encoding UTF8
        Write-Host "Saved: $csv ($($allRows.Count) rows)"
    } else {
        Write-Host "WARNING: 0 rows for group $($grp.Name)"
    }
}

Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
if (Test-Path "C:\Users\pure ev\Music\Term 3\_calce_peek2") {
    Remove-Item "C:\Users\pure ev\Music\Term 3\_calce_peek2" -Recurse -Force
}
Write-Host "`n=== CALCE extraction COMPLETE ==="
