# Benchmark Results

| Controller | Scenario | Success | Status | Total cost | Fuel used | Artifact directory |
| --- | --- | --- | --- | ---: | ---: | --- |
| scripted_hover_descent | attitude_recovery | False | fuel_depletion | 198125.276 | 9000.000 | outputs/scripted_hover_descent/attitude_recovery |
| scripted_hover_descent | crossrange | False | touchdown_off_pad | 158907.543 | 3852.503 | outputs/scripted_hover_descent/crossrange |
| scripted_hover_descent | low_fuel | False | fuel_depletion | 47977.282 | 2250.000 | outputs/scripted_hover_descent/low_fuel |
| scripted_hover_descent | nominal | False | fuel_depletion | 203908.253 | 9000.000 | outputs/scripted_hover_descent/nominal |
| tvlqr_tracking | attitude_recovery | True | touchdown_success | 94808.143 | 1626.399 | outputs/tvlqr_tracking/attitude_recovery |
| tvlqr_tracking | crossrange | False | touchdown_off_pad | 84074.454 | 1338.287 | outputs/tvlqr_tracking/crossrange |
| tvlqr_tracking | low_fuel | False | fuel_depletion | 44581.468 | 2250.000 | outputs/tvlqr_tracking/low_fuel |
| tvlqr_tracking | nominal | True | touchdown_success | 97175.636 | 1694.841 | outputs/tvlqr_tracking/nominal |
