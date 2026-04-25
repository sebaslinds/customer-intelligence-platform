Set-Location "$PSScriptRoot\..\transformations\dbt"
dbt deps
dbt run
dbt test
