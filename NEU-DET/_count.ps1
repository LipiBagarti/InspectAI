foreach ($d in Get-ChildItem 'train\images' -Directory) {
    $c = (Get-ChildItem $d.FullName -File).Count
    Write-Host "TRAIN $($d.Name) : $c"
}
foreach ($d in Get-ChildItem 'validation\images' -Directory) {
    $c = (Get-ChildItem $d.FullName -File).Count
    Write-Host "VAL $($d.Name) : $c"
}
