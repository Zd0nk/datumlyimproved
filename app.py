"""
Datumly — Data-driven FPL Intelligence
=======================================
Data sources:
  1. FPL API (bootstrap-static, fixtures, live GW) — player stats, prices, xG/xA, form, set pieces
  2. football-data.co.uk — betting odds → match probabilities
  3. Club Elo (api.clubelo.com) — dynamic team strength ratings
  4. Custom xPts model — blends all sources + form-weighting + over/underperformance regression

Optimisation:
  - PuLP MILP solver for squad selection (not greedy)
  - Proper constraints: budget, max 3/team, formation, 15-man squad
  - Horizon-aware transfer planner with escalating hit thresholds

UI: Streamlit with 6 tabs
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
import json
import re
import unicodedata
from pulp import (
    LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value,
    PULP_CBC_CMD,
)
from datetime import datetime
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Datumly — FPL Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FPL_BASE = "https://fantasy.premierleague.com/api"
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"

POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

DATUMLY_LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAfQAAAB+CAYAAADSiuOAAAAgnElEQVR4nO3debwk47nA8d+MsYzMmFaqMAhiCXJzmIgwYyfmWgchbspyRexxo7gSVGQTIioucpV9n7m2SkKMnVjHGmSCnBBLLDGWUKWcYRbDLPePp3rSWvc5VdXVy+nzfD+f/pyZPm+99XZ1nXrqfetdQCmllBpAX4+zqN1lUP0b3u4CKKWUUqpxGtCVUkr1q1w711p6Z9OArpRSSnUBDehKKaXqqq6Vay29c2lAV0oppbqABnSllFI11auNay29M2lAV0oppbqABnSllFKfMVAtXGvpnUcDulJKKdUFNKArpZT6lLS1b62ldxYN6EoppVQX0ICulFJqsay1bq2ldw4N6EoppVQXGNbuAiillOpctWrgpV5fY0cH0hq6Ukop1QU0oCullFJdYES7C6CUUkNFX4+zE3BHhk0uLvX6RzarPKq7aA1dKaWU6gJaQ2+jMI4eALbJsenC5DUfmJe85gAfAH1ABITAG8DrwIvA85Zh9jVaZqWUUp1JA/rgNDx5jQCWSbtRGEevAI8CdwNTLcP8oDnFU0op1Woa0IeWtZLXAcBHYRzdAlxsGea97S1WZwnjaBzwVIZNbrIMc8/mlEYppdLRZ+hD1zLAPsA9YRw9FMbR9u0ukFJKqfy0hq4AtgTuDeNoMnCsZZgz21wepVSH0ElkBg+toatKBwHPhnG0absLopRSKhsN6KraqsD9YRzt2e6CKKWUSk8DuqplWeCGMI72b3dBlFJKpaMBXdUzHJgcxtGu7S6IUkqpgWlAV/0ZAVwXxtG67S6IUkqp/mkv98FnpGWYH5X/E8bRMGA5YPnktQqwGTA++blcg/sbDfw2jKPxlmHOazAvpZRSTaIBfZCzDHMRMDN5vYZMiHIbQBhHSwP7AscA4xrYzTjgOOD0BvJQSinVRBrQu1hSo56MPAvfE7gMWCFndieFcTTFMsy3GilTGEerAhskrzWBsUirwlhgFDASmfRmaf41V/0HyNz0bwEvAc8BfwT+ahnmwkbKozpXGEdLAF8HJgFfBdYBxgALgPeBF4AngN9bhvlEjvzXBvYGtgB6ABM5995Dzrd/APcAd1qG+UKjn0epZtOAPkRYhjk1jKMngWuBrXNkMQr4b+D4tBuEcbQCMAHYPHl9hWyPAJZKXqOR4XTjqncRxtHNwBWWYT6aId9y+VYDZmTdroY9wjhalCH9bMswR9X7ZRhHU4E9MuQ30TLMezKkzzO97W2WYe6WMu8DgKsy5H2xZZiLlwgN42g4cAjwQ+ALdbYZidwIbgecGMbRU8CPLcO8PUX5NgNOBnaqk2Tl5NUD7JZscx9wvGWYf07zgZqtr8cZiZy7aW/Q3wZWL/X68xvc75LAO8jjvTTeANYo9fp6490C2iluCLEM803g34G8c7cfFsbR6AzppwG3IBfmbWj8eX41C7nwPxLG0WNhHE0oOH/VYmEcrYksIHQJ9YN5LV8Bbgvj6Mowjpatk/fIMI7OBR6jfjCvZ3vgT2Ec/Srpt9JWpV5/LnBphk3GktycNGgi6YM5wBQN5q2jAX2ISZrh90GaE7MaA3yj2BIVZjwS2L2kqVYNMskN2XSkM2deBwF3Vt94hnG0MvAA8D0gb0AeBpwAXN4h59j5yGOptA4vYJ/fypB2EXBlAftUKWlAH4Isw3wfOCrn5nsXWZaCDQNOBK7ukAtuq2Rp7u9IYRx9DbgLMArIbivgmnJNOowjC7gfKGpK4+8grU5tVer13wBuzLDJjn09zup599fX4yxFtkdB00q9/st596ey04A+RCXPGh/KsenEMI6WLLo8BbOB09pdCJXaKsBNSF+JokwCjgrjaCngVmD9AvMG+EkYRz0F55mHnyHtcODQBva1E9JKl9blDexL5aABfWg7N8c2I2lsCFyrnBDG0ebtLkSLDPYa+iTkGW/Rfo48i2/GYkNLAWc2Id9MSr3+w0CWjnoH9/U4eVuvsjS3zwRuyLkflZMG9KHtNmBuju0aecbZKsPogAuuaqsVgG83Mf+JYRxl6bjXLFlq6asCmadz7utxlkFuvNK6Lum4p1pIA/oQZhnmHGQcb1aNTAU7G3lWejbS/LcFsB7S7DoaGUr5OWBFYGOkA58PvJ5jXxOGyFKwg72GPlgNQ0ZZtFsAvJshfZ7OcbuQ7ZHIFTn2oRqkAV1Nz7HNGhnTvwechzyDW8EyzJ0sw/y+ZZiXW4b5qGWYL1qG+bZlmLMsw1xgGeYcyzBDyzCfsgzzesswjwHWAg4DZmXc9wEZ0w9G3TYs6Fngu8hEMiORmvaOwH0F5f8McjO5FjKRjIWM3ng6R14TCypTbqVefx7yaCGtnfp6nM9n3E2W5vbeUq//ZMb8VQF0Yhn1Wo5t0j7vfAEZWjPZMsyGmt8sw1wAXBbG0QzgDtIPPdoFcOrk+UatfHJMunKTZZh7ZkhftG6qoV8GfK9q3YCPgD+EcXQPcDM5mowrnAccZxnmJxXvzQOmhnF0JzJ3QpZWnQ3DOBphGWZDE7YU4AJkhEeaDqtLIC0LJ6fJuK/HWZZsx1xr522iNXT1To5tPpcmkWWYe1uGeWGjwbwqz7uQIUhprZ3MWNfNuqWGfitwRL1FgJJpfk9sIP8bAKcqmFfm/xF1bv76sQzw5QbKVIhSr/82cH2GTQ7J0DluN1L+zQMfA1dnKIcqkAZ0NSfHNmn/uJsl6zSvGzalFJ2jG2ro84CjBpqb3zLMZ4E3c+Q/Fzg6Wcyov/wfB17JmHfusd0Fy9I5bjVg55RpszS331zq9aMM6VWBNKCrPLNmtTuAvJcxfadccJulG2roUyzDTDuv/l9y5H+FZZhvp0yb9YYxy9jspin1+n8kWyfXATvH9fU4o0gf+EHHnreVPkNXNee9HkCeWv1iYRwthyzashHSw30N5Ln88sh870smr6LmzG7GGOdO0g0BPcuY5TwjHrI0R2etoZcypm8mn/RN3rv09Tirlnr9/lo8dkc6JqbxBvCHlGlVE2hAV3mC3eysGyTPsfdHmu82QzrmtEqem5bBpN0tJo2aj3RGSytrk+484OEM6WdmzL/IGe4a9Vvgf0j3d13uHHdKP2myNLdP1oVY2kub3FWeiTFSP8MM42h0GEdnILWqc5BlVFs9z/oyLd5fqw32gP5yvY5wdXyYMf/nM/ZCz9qJs2Ouo6Ve/xPgogybHNLX49Qsf1+PsxwyXDANXYilA3TMiajaZpMc26RaqS1ZcONZZA31dtaSu/08H+wB/e8Z02cJ/gAvZUzf9uVRG3QR0ts8jdWpv5TsnsDSKfN5oNTrZ31UoQrW7Rc61Y/kWXaegD7gBTKMo62Q4WVZJ7BQQ0/WGveCjOnfz5h+UD+iKfX67wK/ybBJvc5xWZrbdex5B9CAPrTtjiwykVW/PWnDOFoF+D3tH942VOSpUXZS/5mss/9lbZH4IGP6Tjo2eZ2TIe2ufT3OKpVv9PU4y5N+FjxdiKVDaEAf2o7Osc1cZOrM/pwFmDnyVvnk6ZOwXOGlyC9rjTurds/i1nKlXn866YffjQAOrnrvG6SbdQ50IZaOoQF9iArjaC/yLSt5V72ZtpJ81yJbU13Z68AvgR2QZvpRlmEOq/UC/jtH/t0sTytLqehCqI6TZaKZ6s5xWf6Gdex5h+iGpiWVURhHJtn+2CsN1LS2H9mbgM8GTsrQ07nVveQ7XdpxwpU2KLwUqtPcgIwNXy1F2jWRJva7+nqcFYDtU+6jt9Tr/ylf8VTRtIY+xIRxNBJ5vr1qjs37gKkDpEl7ISi7Kll5LUvP5W5vzs86ljfPOOiNc2yjBpFSrz8fuDDDJuXOcXuTvrKntfMOogF9CAnjaHXgXmCrnFlcahnmQB2Ysi5UcXqOcnwxxzaDSdZOYitmSZzc1LV92U/VEpcgq9WlsXtfj7My6ZvbdSGWDqMBfYgI4+ibyHrPE3JmMQv49QD7WBJZWzqtuZZh/i1LIcI4GgFsl2WbHLJ2osrT5N2frDOVrZMx/T501uxmqkmShVKuTZl8BLKa3TYp099U6vWzrqugmkifoXexMI6WQZ5pH0PjK46dlmJxi6zD1Op2ruvH/sic782UtkZTtnbB+88a0DdPmzCMo2WBUzPmrwY3n8/2Yq/nGNL3gdGx5x1GA/ogF8bRMKS2tTzSc3lVZK70CcB4iqmJPY10XBvIbGSMcNoLwnJhHI1NuwpWGEcl4OSUeTci61zha4dxtI1lmFnmI+9P2lXHyjYN42izZOnPupJz5UK6f/U5VaHU6z/T1+M8CGydInnav11diKUDaUAffOaGcUuXG/4Q+A/LMAecStIyzE/COOojWw36UFLUGMM4Wgq4BumN21SWYfaFcTQLGJVhs9+HceQhF7lXgVkDre3dj6y9hocBvwvjaFvLMGtOvxnG0SjgXODAnGVSg9s5pAvoaelCLB1IA7rqz3xgX8sws8yF/SrZAvqPwjh60jLMO+slSGaeuwbYNkO+jXqebNPiGsAZyQuAOjde71mGOVAv/b8g85WnnUcbZOz+02EcnYksFfpqsv3qwCTgCHQa3qHsJmQNhjUKyEsXYulQ2ilO1bMQOMgyzNsybpe1GW5p4PYwjqaEcbRDGEcrhHE0IowjK4yjrcI4Oht4kdYGc0g/y1bhkol7Hsqx6Wjg58iCOHOQOcyfAX6BBvMhrdTrLwAuKCg7XYilQ2lAV7XMAfa2DPOaHNvenGObYUhT8N3I8+tPgHeBB5FZ4doxJ3zWG5miXdbm/avucynyt90oHXveoTSgq2ozgG0sw5yaZ2PLMB8DHiiyQG1yLymXiW2SG4GwjftXXabU679P4+PGZyITU6kOpAFdVboc+LJlmI1O5Xgi2Wc7S2s2MLlJeS9mGeYCWtOjvt7+PwZOaFL2C4FfNSlv1dnyTvlcpguxdDAN6AqkRr21ZZiHWoaZdanJz7AM8wngBw2X6rM+RmaxGmi1t6JMQWrKbWEZ5mSKb/qfDxwCBAXnqwaBUq//LHBfA1loc3sH04A+dM0BrkMC+XaWYebphFWXZZi/Bs4sMMtZwO45OunlZhnmImRinnYGvwOBPxaU10zkGE4uKD81OGVZK72SLsTS4TSgDx2LgJeQ5ur9gBUtw9yv6EBeyTLM45GA1GhHnKeATSzDvKvxUmVjGeZHlmHuC+yFTLDT6v3HyJKytzaY1TRgnGWYdzReKjXI3Qrk6aWutfMOp+PQB68FyWseMBcJmh8gK6JFwDvAm0jHrheBv1mG+WGrC2kZ5lVhHD0EuMBBZBtb/TxwFnBFA5O0FMIyzBuBG8M4+jKwE/A1YD1gLDAGWZM867Kxafc9G5gUxtEkwAO+lGHz6YBnGeb1zSibGnxKvf7Cvh7nfORvKy1diGUQaMoFSKlawjhaEQmG2wPjkIVcVkBaimYD/0RaEZ4A7h5oKtOhKIyj4cBXgR2RsfmrIsvJLo/c2IXIjdCjwG2WYT7VnpKqTtbX44xBpm9NOxvi70q9/n80sUiqABrQlVJqCOrrcaYCe6RMvnOp1687m6PqDPoMXSmlhpi+HmdJYMuUyXUhlkFCA7pSSg09OyOPu9K4UhdiGRw0oCul1BDS1+MMB36UMvkC4JImFkcVSAO6UkoNLR6wacq0N5V6/TeaWRhVHB22ppRSXaqvxxkGjERGQ2wKHE62ddHzTkKj2kADulJKdYm+HmccMhFTEe4q9foPFpSXagFtcldKKVVtETIZlBpENKArpZSqdnap13+63YVQ2WhAV0opVenPwEntLoTKTgO6UkqpsheBSaVe/+N2F0RlpwFdKaUUyPz/25Z6/bfaXRCVjwZ0pZQa2mYARwNblXr9t9tdGJWfDltTSqmhYxayxPJrwOPAfcCdOrVrd9DV1gYpzw+WQZbLrDQPmAm8iixBeq3r2H9sddmazfODCJjlOvaaGbZZDamJ3OQ69p5NKlpT1fnOq+3rOnZQJ+1C4D3gSeAc17EXL7hRkX6m69ilBsrYkce5Xrlqvd+pn0GpgWiTe3dZGlgR2AxpQnvM84ObPT9IuwhDXZ4frOP5wSLPD4JG81JtMxxZg34X4C7PD/6rzeWpS883pbLTJvfBb3GNyvODJYDlgX8D9kameZwE3Or5wTauYw/ZnquuY79B97RIZalFV54fSwJfBE5D1sH2PD+4ynXsD4oqWDcc5274DGpo0oDeRVzHXgBEwDRgmucH/wfcD4wHDgUuaGPxVJu5jv0J8KznB99CnqGuDEwA7mpnuZRSxdCA3sVcx/6T5wenAGdQFdA9P9g4eW9r4AvAJ8AzgO869g2V+Xh+4AKnJ//9VhIQyv7Tdeyrs+Y5EM8PhgMOcESS1zvAtcDJddKPQ+awngL8LCnvDoAJbA68QcVzUc8PJiDDdG50HXuvOnn+DVgLGOs6dlzx/gTgB8AWgAGEwN3AKa5jv9JPuX6IfBc7A58D/gr8zHXs21MfmAK4jj3P84PnkYBuFpl3nWfS48hwDNKeb0na1N9FI5+h4ndLAMcChwFr8unz8k1q9O1oxfni+cF44Dhgy2QfM5Cb+V+5jv1y3vKowUWfoXe/a5KfG3l+MLri/enAd5Hm+WWBMUggvt7zg+Ny7qvIPC8Efg2sj/QNWB2ZW/oG+m8OXRnpvbsv8ry4ZlrXsR8DXgB2q9XHwPODTZN931IVzA8DHgb2AlYClgRWAb4NTPf84Mt1yrViUq4DgBWAZYBNgFs8P9iqn89TOM8PlgbWS/4btXDXhR6DBr6LRlwCnIkcvwHPy1acL54fHAk8AuwDjE3KtQ5y03F5QeVRg4AG9C7nOvZbQIx812MrfvU4sD+wLnKxGAscAnwInOr5QakiDy9JB/Ab17GHVbyuzpNnfzw/2BZ5/j8X6dw3FgnU/wVsh9Qq6tkReBfYHhidlLFeT/8pyAVt3xq/+3ZFmnK5NgDOT/L/NnIhXAZYG7nIl5AbkVp2RmpC2wKjkdrdb5Dv5fv9fJ5axiQdxqpfd/a3kecHS3p+8CWkRjkWmA08lnHfjUh1DNKcbw1+F7l4fvB14GDkuB2FHMOVkHN1G6rOy1acL54fbAich9xMXAZsCIxK9nE48PeCyqMGAW1yHxo+RC42i2vormOPr0rzT+AKzw8swEOa7m7NspMC8zwo+flT17HPq3j/gqRj1//2s+1cYFfXsWek2M9VwC+Qi9vi/Xh+sBRgIxe+OyrSH4XcAOzjOvbDFe+/AhyfXDB39fxgJdex36naVx+ws+vY7yb/n+X5waHA7siohGYZ4/nBojq/O7HIDnEp9FHcMWjku8jrwOTnj1zHrgx8lybn5fkFlrGPdMfqu8ASwIWuYx9VtY9XgEsLKo8aBDSgDw3LJT8XX7w9P1gbOBGpya6K3KlXWi3rTtLm6fnBCOT5erWtkgvNV5L/X1UjzRT6D+iPpwzmuI79hucH9wITPT/4kuvYzyW/moTcAP3adez5FZtMSH4+4PkB/KuJdRifbm5dA3m2WunRiotzef+zPD94DakhZZF3rPhCpLWmPA691Z3hijwGjXwXeZXPy+tq/O4aPhvQW3G+lG+iK29862nHMVMtpAG9y3l+sAoylG0B8Hby3peQDmFj+tm0OhgPtJ8i8xwDzK9VS3Adu8/zgzn9bJsqmFeYDExEauknJu99prk9UX7WvsQAeS5V472wTtqPU+TXiIYmiilYkcegke8ir+WQ8/Ld6l+4jj2zxnnZivOllPx8bYB9NFoeNQhoQO9+ByQ/n3Yde1by7xOQoHkVcBbS5DbbdeyFSQebPM/RUueZ1Hr769g2E1ijVtNf8hx+2X62zTqF5Y1Iy8UBnh+chNTMdwaecR37mRrlAljDdezXM+5HFasd38UHyHm5YnVQ9/xgDJ89L1tRxr7k55rAc/WTtaw8qo20U1wX8/xgE+AnyX8rn6Wtk/w81nXsZ1zH/tB17HIg3K1OduWm56Xr/D5PnvU8lfzcv8bvDqzxXm6uY88Ffot0ENoh2ecIPls7Byh3rjuyyDKomgY639rxXTyd/LRr/G6/Gu+1oozlfXwvQ1o9f7uU1tC7SDJGtsS/Zoo7ArkgPgJcWZH0H8gY1J96fnAG8lx1XeB4YNc62b8LLAI28/xgXeDlioCdN896piDN3qd6fvARMiRoEfAN4JcZ80pjMjJ+/kBgAySYXFMj3blIr33X84PlkCFMryDHeE1gJ2BT17H3aEIZh5qBzrd2fBf/B/wncFpyXk5Nyrg78Ksa6VtRxguQ4WlHJn1TzgVeRnrffx0Y7zr2oS0sj2ojraEPfouHMCGBqDxTnIP8od4E7F417ev5SNP0MchkGHOBvyC1jFo1U1zHnpPkOxZ4EViQ7LfcpJ85z3pcx74faVFYNsn3n0gnnYuSMsT1t87OdexHkOE9+yAdn+6o85z0WSTwz0eG0D2DjCCIgD8hPeazdu7qNPWGxJVfmfpW5DXQ+daO78J17HuQc3kUcDFyTr6LDBd7EGn+/qQifdPL6Dp2LzK0cxES2P+CDKsr93BfpyLtUDh/hzQN6N3lY6QzzRPI3fh417H3rJwYBcB17EeRu/FHkOeCM4F7kDHe9/ST/0HIM+cYuYAUkWc9RyIzX72YfK43kLGye1fvuyDlMenlf9fkOvYU4GtIbe11/nXMnwROQXrIq2IcRJ3zDdr2XRyC9Bd5KdnfDKR2/h2kdaz6b63pZUyG0G2NtBiEyKqLLyE3HQe3ujyqfXQBAqWUapDnBwcjs7Jd6jr24e0ujxqa9Bm6Ukql5PnBj4BZyII2/0CeVe8OnJok0eVeVdtoQFdKqfTGIs+fa5niOvZ9rSyMUpU0oCulVHq/QJ5R/zvSMxzgb8gokovbVCallFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkqpz9C53FVTeX6wAuAi02OugSzc8ndkedIrk1W1OoLnB18B/gw86jr2Fi3etwmchByn1ZAVsJ4CznId+wHPD8YAP0OWkF0JWU3rN4DvOvbMJI+JyIp0o4EzXMc+syL/lZAFOLZ0Hfv1ln2wfnh+MBlZJhdkBbAZwO+An7uOPcfzg1uBv7uOfWx7SvhpVeWttJXr2A/X+DzvA38Ffgtc5jr2/Br5zEcWHroR+Knr2LOaVX7V/XS1NdU0nh+shgTITZBVqpYH1geOBdZDllbtJIchQW+85wcbtGqnnh+sCkwHxgEHIKt2fQ1ZOvYnyTr3VyHrW38TWCH5ORxZ5QvPD5ZO0nwf2BI4yfODjSp2cy5wZqcE8wrTXMcehiyVeyhwBLXXFu8U01zHHlb1erj698jSxRsiC7acBNxXtfRsOd1IZI31A4EzWvQZVJfSqV9VLp4fWMjayxe6jn1K8t6GyNKtB7iOfT0SRAB2dh37o+Tfc5M0T3RIGctpRyI3GPshNxyHAD9odhkT5eO0S8Vxejt53ZkE652BQ1zHnp78/nng5xV5rAsMcx17KoDnB/cggf0Zzw8mAasD5zX1U1TIcuwBXMf+BAl6U4A9PT8YDeyabHdMkmwD17Gf9/xgR+CXyM3hW8i632e5jr2gBR8tFdexFwL/BK7x/OAx4DnAoSpoJ7X2hz0/uArYq+UFVV1Fa+gqF9exQ2S96h97fjAhCYjXAde5jn19ckHeDbigIkh1VBmrkn8TeRxwJ3AJcKDnB0vSZJ4fjELWoO7vOH0MzAa2y1qm5Hv4X+DQJMi0RMZjX2v7g4DbgHMqasLPJ8frRmTudAuYiLRo9DTlgxTAdexXgDuAvftJVq7VK5Wb1tBVbq5j3+X5wQXI8/BpyAXp6OTXayHn14tF7c/zg5OBY13HLhVUxkqHApe7jr3Q84ObkNrs7sANzSwfKY6T69iLPD84CrnRmOT5wSPAI8BU17HL270EDPP8YA/gWWAHpBbrAdcCYzw/eB4wkefupzT5c2U59nh+MALYAml6vrafbFdEmqlvTvpfvIY0aaeW9/MktvH8YFHF/6e7jr1Jiu1eQD5fdVlGAJshj1quzlEepRbTgK4adSKwE3Ih3rxGp57Kix+eH0TIM2CA21zH3q35Rey/jJ4frINcbA8Aaf71/OBKJMinDug5lTumLuovkevY13p+cBewI7B5UrbTPD9wXMe+0HXseZ4fHAhcgHSK85DAtx3Sh+FZ4MfAw8Djnh/c7Tr2Y035RJ820PlRDpDlzmFXAD/tJ79XkdruY54fBMD9wL2uY88tvOS1TXMde9uc21Z+x9U3BlNp3SMe1aU0oKtGrQl8HrlYrQU8nrz/CnKR/mJlYtexTQDPD64HKjsJDch17JOBkwssY9mhwBLA654fVL6/0PODz7uOPaOJ5Ssfp/VS5P8eUnu91vOD4UjwO9Pzg0tcx17gOvadyOfD84OlkI52RyBN05br2Nckv7sd2BpIFdAbOO4w8LHPFCCT1opdkfJPBE4DLvb8YKLr2M+lzONk8n+evNZDbkbKprmOvW3SP2Iv5BHCnsCAjyOUqkefoavckue51wA3I7WLCz0/WB3AdewPkWeg300uWh1XxuT3I5AhRHZ172WkNvudZpYvOU63AkdV9YIeaLuFSPlGJq9qPwQecR37Ido0PHWgY5/CJ9S4RrmOvch17GmuY//YdeyNkI5xBxdR5mbw/OALSKfG31f/znXsea5jXwecDZyf9HlQKhetoatGnIo809wBmIk0rV7l+cF2ScA5GngUeNDzgx8CzyC93NcD1kaaWNtdxl0BA2nGrTYVOMbzg180uUOZgxyn25Pj9DQyxG8cEgh3T8p3DtJL/D1gI+A4pLm5+hHCBkiHtHHJWzOA2POD/ZGbgF2Q4VTNNtCxH8g/gI09PxhV/oyeH4xHbrIuQJ5Lr4+M23+5CeXPLWlBsYDtgdOR1pJz+9nkdKSl6ATgJ00voOpKWkNXuXh+sA0y5vlA17H7XMdehASRDZDnpiRN1RsjQeQi4E0kuFwG3IKM+25rGZHhafe5jv1BjSxuRCbD2aGZ5UyO01eRYV7XIr3tpwPfA05LOn/9GLCRZvL3kJrvH4BvVebl+cEwZBjXceUJZ5LhXAciE9P8GbjIdexHmvmZUh77gZyDXKPe8fxgkecH6yPzBEwHpgAx8h1dAlxc7CfIrfxs/GNkUpnDgf8Btu3vOX/SUnMqcJznB2NbUlKllFJKKaWUUkoppZRSSimllFJKKaWUUkqpTvX/lx8/Oo3jlMsAAAAASUVORK5CYII="
POS_FULL = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}

# FPL points system constants
PTS_GOAL = {1: 10, 2: 6, 3: 5, 4: 4}
PTS_ASSIST = 3
PTS_CS = {1: 4, 2: 4, 3: 1, 4: 0}
PTS_APPEARANCE = 2  # 60+ mins
PTS_BONUS_AVG = 0.5  # average bonus per appearance

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0a0e17; }
    header[data-testid="stHeader"] { background-color: rgba(10,14,23,0.9); backdrop-filter: blur(10px); }

    .main-title {
        background: linear-gradient(135deg, #f02d6e, #e8456e, #ff6b8a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 800; letter-spacing: -1px; margin-bottom: 0;
    }
    .sub-title { color: #8892a8; font-size: 0.85rem; margin-top: -8px; }

    .metric-card {
        background: #111827; border: 1px solid #2a3550;
        border-radius: 14px; padding: 1.1rem; text-align: center;
    }
    .metric-label { color: #5a6580; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
    .metric-value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; }
    .metric-sub { color: #8892a8; font-size: 0.75rem; margin-top: 2px; }

    .fdr-1 { background:#065f46; color:#6ee7b7; padding:2px 7px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-2 { background:#14532d; color:#86efac; padding:2px 7px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-3 { background:#78350f; color:#fcd34d; padding:2px 7px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-4 { background:#7c2d12; color:#fdba74; padding:2px 7px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-5 { background:#7f1d1d; color:#fca5a5; padding:2px 7px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; margin:1px; }

    .transfer-card {
        background: #1a2236; border: 1px solid #2a3550;
        border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.4rem;
    }
    .transfer-out { color: #f87171; font-weight: 600; }
    .transfer-in { color: #34d399; font-weight: 600; }
    .transfer-arrow { color: #38bdf8; font-size: 1.1rem; }

    .gw-bar {
        background: #111827; border: 1px solid #2a3550; border-radius: 12px;
        padding: 0.65rem 1.1rem; display: flex; align-items: center; gap: 1rem;
        margin-bottom: 1rem; flex-wrap: wrap;
    }
    .gw-num { background: linear-gradient(135deg,#f02d6e,#ff6b8a); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.05rem; font-weight: 700; }
    .gw-deadline { color: #8892a8; font-size: 0.78rem; }

    .badge { font-size:0.65rem; padding:3px 9px; border-radius:6px; font-weight:600; }
    .badge-green { background:rgba(52,211,153,0.15); color:#34d399; }
    .badge-yellow { background:rgba(251,191,36,0.15); color:#fbbf24; }
    .badge-blue { background:rgba(56,189,248,0.15); color:#38bdf8; }

    .pitch-row-label { color:#5a6580; font-size:0.68rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:5px; margin-top:12px; }
    .pitch-shirt { width:40px; height:40px; border-radius:10px; display:inline-flex; align-items:center; justify-content:center; font-weight:700; font-size:0.75rem; color:white; margin:0 auto; }
    .pitch-shirt-gkp { background:#f59e0b; }
    .pitch-shirt-def { background:#3b82f6; }
    .pitch-shirt-mid { background:#10b981; }
    .pitch-shirt-fwd { background:#ef4444; }
    .pitch-name { font-size:0.7rem; font-weight:600; color:#e2e8f0; margin-top:3px; }
    .pitch-price { font-size:0.58rem; color:#5a6580; }

    .section-header { font-size:1rem; font-weight:700; margin-bottom:0.5rem; margin-top:1rem; }

    .source-tag {
        display:inline-block; font-size:0.6rem; padding:2px 6px; border-radius:4px;
        font-weight:600; margin-left:6px;
    }
    .src-fpl { background:rgba(56,189,248,0.15); color:#38bdf8; }
    .src-odds { background:rgba(251,191,36,0.15); color:#fbbf24; }
    .src-model { background:rgba(167,139,250,0.15); color:#a78bfa; }

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LAYER
# ============================================================

@st.cache_data(ttl=3600)
def load_fpl_data():
    """Source 1: FPL API — player stats, prices, fixtures, xG/xA."""
    try:
        headers = {"User-Agent": "FPL-Optimizer/2.0"}
        b = requests.get(f"{FPL_BASE}/bootstrap-static/", headers=headers, timeout=30).json()
        f = requests.get(f"{FPL_BASE}/fixtures/", headers=headers, timeout=30).json()
        return b, f, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data(ttl=7200)
def load_betting_odds():
    """Source 2: football-data.co.uk — match betting odds for PL 2025-26."""
    try:
        resp = requests.get(FOOTBALL_DATA_URL, timeout=20)
        if resp.status_code != 200:
            return None, "HTTP " + str(resp.status_code)
        df = pd.read_csv(StringIO(resp.text), on_bad_lines="skip")
        if len(df) == 0:
            return None, "Empty CSV"
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=7200)
def load_club_elo():
    """Source 3: Club Elo ratings — dynamic team strength from clubelo.com."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        resp = requests.get(f"http://api.clubelo.com/{today}", timeout=15)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        df = pd.read_csv(StringIO(resp.text), sep=",")
        if len(df) == 0:
            return None, "Empty response"
        # Filter to English teams only (Country == ENG, Level 1)
        eng = df[(df["Country"] == "ENG") & (df["Level"] == 1)].copy()
        # Build lookup: club name -> Elo rating
        elo_map = {}
        for _, row in eng.iterrows():
            elo_map[row["Club"]] = float(row["Elo"])
        return elo_map, None
    except Exception as e:
        return None, str(e)


# Club Elo name -> FPL short name mapping
ELO_NAME_MAP = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Chelsea": "CHE",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Ipswich": "IPS", "Leicester": "LEI", "Liverpool": "LIV",
    "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
    "Nottingham Forest": "NFO", "Southampton": "SOU", "Tottenham": "TOT",
    "West Ham": "WHU", "Wolves": "WOL",
    "Leeds": "LEE", "Burnley": "BUR", "Sunderland": "SUN",
    "Sheffield United": "SHU", "Norwich": "NOR",
    "Middlesbrough": "MID", "Luton": "LUT",
}


@st.cache_data(ttl=3600)
def load_recent_gw_live_data(current_gw_id, n_recent=7):
    """
    Fetch per-GW live stats for the last N completed gameweeks.
    Uses event/{gw}/live/ endpoint — one call per GW, returns all players.
    Returns: dict {player_id: [{gw, minutes, xG, xA, xGC, goals, assists, ...}, ...]}
    """
    headers = {"User-Agent": "FPL-Optimizer/2.0"}
    player_gw_data = {}

    start_gw = max(1, current_gw_id - n_recent)
    for gw in range(start_gw, current_gw_id):
        try:
            resp = requests.get(
                f"{FPL_BASE}/event/{gw}/live/",
                headers=headers, timeout=15,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            elements = data.get("elements", [])
            for el in elements:
                pid = el["id"]
                stats = el.get("stats", {})
                if stats.get("minutes", 0) == 0:
                    continue
                if pid not in player_gw_data:
                    player_gw_data[pid] = []
                player_gw_data[pid].append({
                    "gw": gw,
                    "minutes": stats.get("minutes", 0),
                    "goals": stats.get("goals_scored", 0),
                    "assists": stats.get("assists", 0),
                    "xG": float(stats.get("expected_goals", 0) or 0),
                    "xA": float(stats.get("expected_assists", 0) or 0),
                    "xGC": float(stats.get("expected_goals_conceded", 0) or 0),
                    "xGI": float(stats.get("expected_goal_involvements", 0) or 0),
                    "clean_sheets": stats.get("clean_sheets", 0),
                    "bonus": stats.get("bonus", 0),
                    "total_points": stats.get("total_points", 0),
                })
        except Exception:
            continue

    return player_gw_data


def compute_form_weighted_xg(player_gw_data, n_recent=7):
    """
    Compute form-weighted xG/90 and xA/90 from recent GW data.
    Uses exponential decay: most recent GW has highest weight.
    Returns: dict {player_id: {xg_form_per90, xa_form_per90, xgc_form_per90, form_minutes}}
    """
    result = {}
    decay_factor = 0.85  # each older GW is worth 85% of the next

    for pid, gw_list in player_gw_data.items():
        # Sort by GW descending (most recent first)
        sorted_gws = sorted(gw_list, key=lambda x: x["gw"], reverse=True)[:n_recent]

        if not sorted_gws:
            continue

        weighted_xg = 0
        weighted_xa = 0
        weighted_xgc = 0
        weighted_mins = 0
        total_weight = 0

        for i, gw_data in enumerate(sorted_gws):
            weight = decay_factor ** i  # most recent = 1.0, then 0.85, 0.72, 0.61...
            mins = gw_data["minutes"]
            if mins > 0:
                weighted_xg += gw_data["xG"] * weight
                weighted_xa += gw_data["xA"] * weight
                weighted_xgc += gw_data["xGC"] * weight
                weighted_mins += mins * weight
                total_weight += weight

        if weighted_mins > 0 and total_weight > 0:
            nineties = weighted_mins / 90.0
            result[pid] = {
                "xg_form_per90": weighted_xg / nineties,
                "xa_form_per90": weighted_xa / nineties,
                "xgc_form_per90": weighted_xgc / nineties,
                "form_minutes": weighted_mins / total_weight,  # avg weighted mins
                "form_gws": len(sorted_gws),
            }

    return result


def detect_blank_double_gws(fixtures, planning_gw_id, n_gws=6, teams=None):
    """
    Detect blank and double gameweeks from fixture data.
    Returns: dict {team_id: {gw: fixture_count}} where 0 = blank, 2+ = double
    """
    if teams is None:
        teams = {}

    team_fixture_counts = {}
    for t_id in teams:
        team_fixture_counts[t_id] = {}
        for gw in range(planning_gw_id, planning_gw_id + n_gws):
            team_fixture_counts[t_id][gw] = 0

    for f in fixtures:
        ev = f.get("event")
        if ev and planning_gw_id <= ev < planning_gw_id + n_gws:
            if f["team_h"] in team_fixture_counts:
                team_fixture_counts[f["team_h"]][ev] = team_fixture_counts[f["team_h"]].get(ev, 0) + 1
            if f["team_a"] in team_fixture_counts:
                team_fixture_counts[f["team_a"]][ev] = team_fixture_counts[f["team_a"]].get(ev, 0) + 1

    return team_fixture_counts


def odds_to_probabilities(odds_df, teams_map):
    """
    Convert betting odds to match probabilities per team.
    Returns dict: team_name -> {attack_strength, defence_strength, cs_prob, goal_expectation}
    """
    if odds_df is None or len(odds_df) == 0:
        return {}

    # Standardise column names
    cols = odds_df.columns.tolist()
    # We need: HomeTeam, AwayTeam, B365H, B365D, B365A, FTHG, FTAG
    required = ["HomeTeam", "AwayTeam"]
    if not all(c in cols for c in required):
        return {}

    # Use B365 odds (most common), fallback to average odds
    h_col = "B365H" if "B365H" in cols else ("AvgH" if "AvgH" in cols else None)
    d_col = "B365D" if "B365D" in cols else ("AvgD" if "AvgD" in cols else None)
    a_col = "B365A" if "B365A" in cols else ("AvgA" if "AvgA" in cols else None)

    if not all([h_col, d_col, a_col]):
        return {}

    team_stats = {}

    # Process each team's matches
    all_teams = set(odds_df["HomeTeam"].unique()) | set(odds_df["AwayTeam"].unique())

    for team in all_teams:
        home_matches = odds_df[odds_df["HomeTeam"] == team].copy()
        away_matches = odds_df[odds_df["AwayTeam"] == team].copy()

        # Calculate implied probabilities from odds (with overround removal)
        win_probs, cs_probs, goals_for, goals_against = [], [], [], []

        for _, m in home_matches.iterrows():
            try:
                h, d, a = float(m[h_col]), float(m[d_col]), float(m[a_col])
                overround = (1/h + 1/d + 1/a)
                win_probs.append((1/h) / overround)
                # Clean sheet proxy: P(win)*0.35 + P(draw)*0.55 (approx CS given result)
                p_win = (1/h) / overround
                p_draw = (1/d) / overround
                cs_probs.append(p_win * 0.35 + p_draw * 0.55)
            except (ValueError, ZeroDivisionError):
                continue
            # Actual goals if available
            if "FTHG" in m and pd.notna(m.get("FTHG")):
                try:
                    goals_for.append(float(m["FTHG"]))
                    goals_against.append(float(m["FTAG"]))
                except (ValueError, TypeError):
                    pass

        for _, m in away_matches.iterrows():
            try:
                h, d, a = float(m[h_col]), float(m[d_col]), float(m[a_col])
                overround = (1/h + 1/d + 1/a)
                win_probs.append((1/a) / overround)
                p_win = (1/a) / overround
                p_draw = (1/d) / overround
                cs_probs.append(p_win * 0.30 + p_draw * 0.55)
            except (ValueError, ZeroDivisionError):
                continue
            if "FTAG" in m and pd.notna(m.get("FTAG")):
                try:
                    goals_for.append(float(m["FTAG"]))
                    goals_against.append(float(m["FTHG"]))
                except (ValueError, TypeError):
                    pass

        if win_probs:
            avg_gf = np.mean(goals_for) if goals_for else 1.3
            avg_ga = np.mean(goals_against) if goals_against else 1.3
            team_stats[team] = {
                "win_prob": np.mean(win_probs),
                "cs_prob": np.mean(cs_probs),
                "avg_goals_for": avg_gf,
                "avg_goals_against": avg_ga,
                "attack_strength": avg_gf / 1.3,  # relative to league average
                "defence_strength": avg_ga / 1.3,
            }

    return team_stats


# Team name mapping: football-data names → FPL short names
TEAM_NAME_MAP = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Chelsea": "CHE",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Ipswich": "IPS", "Leicester": "LEI", "Liverpool": "LIV",
    "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Southampton": "SOU", "Spurs": "TOT",
    "West Ham": "WHU", "Wolves": "WOL",
    # 2025-26 promoted teams (adjust as needed)
    "Leeds": "LEE", "Burnley": "BUR", "Sunderland": "SUN",
    "Sheffield Utd": "SHU", "Norwich": "NOR", "Middlesbrough": "MID",
    "Luton": "LUT",
}


def build_xpts_model(players_df, team_odds, teams_map, fixtures, current_gw_id,
                     form_xg_data=None, team_fixture_counts=None, elo_ratings=None):
    """
    Source 3: Custom expected points model.

    For each player, for the next N gameweeks, estimate xPts by combining:
      - FPL API xG/xA per 90
      - Betting odds (team attack/defence strength, CS probability)
      - Playing time probability
      - FPL scoring system

    Returns: dict player_id -> {gw: xpts, ...} for next 6 GWs
    """
    # Build FPL team short_name -> odds stats mapping
    odds_by_fpl = {}
    for odds_name, fpl_short in TEAM_NAME_MAP.items():
        if odds_name in team_odds:
            odds_by_fpl[fpl_short] = team_odds[odds_name]

    # Build FPL short_name -> Elo rating mapping
    elo_by_fpl = {}
    if elo_ratings:
        for elo_name, fpl_short in ELO_NAME_MAP.items():
            if elo_name in elo_ratings:
                elo_by_fpl[fpl_short] = elo_ratings[elo_name]

    # League average Elo (for normalisation)
    if elo_by_fpl:
        avg_elo = np.mean(list(elo_by_fpl.values()))
    else:
        avg_elo = 1600  # default PL average

    # Build opponent map per team per GW
    upcoming = {}  # team_id -> [{gw, opp_id, home, difficulty}]
    for t_id in teams_map:
        upcoming[t_id] = []
    for f in fixtures:
        ev = f.get("event")
        if ev and current_gw_id <= ev < current_gw_id + 6:
            if f["team_h"] in upcoming:
                upcoming[f["team_h"]].append({
                    "gw": ev, "opp_id": f["team_a"], "home": True,
                    "difficulty": f.get("team_h_difficulty", 3)
                })
            if f["team_a"] in upcoming:
                upcoming[f["team_a"]].append({
                    "gw": ev, "opp_id": f["team_h"], "home": False,
                    "difficulty": f.get("team_a_difficulty", 3)
                })

    # League average goals per game (approx)
    league_avg_goals = 1.35

    xpts_all = {}
    xpts_breakdown = {}  # {pid: {gw: {component: value}}}

    for _, p in players_df.iterrows():
        pid = p["id"]
        pos = p["pos_id"]
        team_short = p["team"]
        mins = p["minutes"]
        starts = max(p.get("starts", 0), 0)
        total_gws_played = max(current_gw_id - 1, 1)  # GWs elapsed so far

        # ============================================================
        # EXPECTED MINUTES MODEL
        # ============================================================
        # Key insight: a player's expected minutes next GW should be based on
        # their actual average minutes per GW this season, not just a binary
        # "available or not". This prevents fringe/youth players being inflated.

        # Average minutes per GW this season
        avg_mins_per_gw = mins / total_gws_played

        # FPL's chance_of_playing (None = no news = likely available)
        chance = p.get("chance_playing", None)
        if chance is not None and not pd.isna(chance):
            availability = float(chance) / 100.0
        else:
            # No news — availability based on recent playing pattern
            if avg_mins_per_gw >= 60:
                availability = 0.95  # regular starter
            elif avg_mins_per_gw >= 30:
                availability = 0.75  # rotation / sub risk
            elif avg_mins_per_gw >= 10:
                availability = 0.40  # mainly a sub
            elif mins > 0:
                availability = 0.15  # fringe player
            else:
                availability = 0.0   # hasn't played

        # Expected minutes next GW (capped at 90)
        # Blend: recent avg mins * availability
        expected_mins = min(avg_mins_per_gw * availability, 90)

        # Convert to "expected 90s" for scaling xG/xA
        expected_90s = expected_mins / 90.0

        # Playing probability (for appearance points — did they get on the pitch?)
        play_prob = min(availability, 0.98)

        # Full 60+ min probability (for clean sheet, appearance pts)
        full_game_prob = expected_mins / 90.0 if expected_mins >= 45 else expected_mins / 180.0

        # ============================================================
        # PER-90 STATS — blend season average with form-weighted recent
        # ============================================================
        mins_played = max(mins, 1)
        nineties = mins_played / 90.0

        # Season-average xG/xA from FPL API
        season_xg = float(p.get("xg_per90", 0) or 0)
        season_xa = float(p.get("xa_per90", 0) or 0)

        # Fallback: use actual goals/assists per 90 ONLY if enough minutes
        if season_xg == 0 and p["goals"] > 0 and mins >= 270:
            season_xg = p["goals"] / nineties
        if season_xa == 0 and p["assists"] > 0 and mins >= 270:
            season_xa = p["assists"] / nineties

        # Form-weighted xG/xA from recent 7 GWs (if available)
        form_data = (form_xg_data or {}).get(pid)
        if form_data and form_data.get("form_gws", 0) >= 3:
            form_xg = form_data["xg_form_per90"]
            form_xa = form_data["xa_form_per90"]
            # Blend: 60% recent form, 40% season average
            # (recent form is more predictive but noisier)
            xg_per90 = form_xg * 0.6 + season_xg * 0.4
            xa_per90 = form_xa * 0.6 + season_xa * 0.4
        else:
            xg_per90 = season_xg
            xa_per90 = season_xa

        # Also get form-weighted xGC for GKs/DEFs
        form_xgc = None
        if form_data and form_data.get("form_gws", 0) >= 3:
            form_xgc = form_data.get("xgc_form_per90")

        # Apply regression to the mean for low-sample players
        sample_weight = min(nineties / 10.0, 1.0)
        pos_avg_xg = {1: 0.0, 2: 0.02, 3: 0.12, 4: 0.35}
        pos_avg_xa = {1: 0.01, 2: 0.05, 3: 0.10, 4: 0.12}
        xg_per90 = xg_per90 * sample_weight + pos_avg_xg.get(pos, 0.1) * (1 - sample_weight)
        xa_per90 = xa_per90 * sample_weight + pos_avg_xa.get(pos, 0.08) * (1 - sample_weight)

        # ============================================================
        # OVER/UNDERPERFORMANCE REGRESSION
        # ============================================================
        # If a player has scored significantly more/fewer goals than their xG,
        # they're likely to regress. Adjust xG towards actual performance mean.
        # E.g., player with 12 goals from 8.0 xG is overperforming — reduce projected xG
        xg_total = float(p.get("xg_total", 0) or 0)
        actual_goals = p["goals"]
        if xg_total > 0 and nineties >= 5:
            overperformance = (actual_goals - xg_total) / max(nineties, 1)  # per 90
            # Apply 30% regression towards xG (don't fully regress — some players are genuinely clinical)
            regression_factor = 0.30
            xg_per90 -= overperformance * regression_factor

        xa_total = float(p.get("xa_total", 0) or 0)
        actual_assists = p["assists"]
        if xa_total > 0 and nineties >= 5:
            xa_overperf = (actual_assists - xa_total) / max(nineties, 1)
            xa_per90 -= xa_overperf * 0.25  # assists regress less aggressively

        # Floor at 0
        xg_per90 = max(xg_per90, 0)
        xa_per90 = max(xa_per90, 0)

        # ============================================================
        # SET PIECE TAKER BONUS
        # ============================================================
        # NOTE on penalties: FPL's expected_goals_per_90 ALREADY includes
        # penalties taken this season. So a pen taker's xG/90 naturally
        # reflects their penalty duty. We do NOT add a separate pen xG boost
        # to avoid double-counting.
        #
        # However, we keep a small boost (+0.015) as a forward-looking signal
        # for penalty ORDER — the FPL API confirms who is on pens even if
        # they haven't taken many yet this season.
        #
        # Corner/FK takers get an xA boost (more delivery opportunities)
        # and direct FK takers get a small xG boost.
        pen_order = int(p.get("penalties_order", 0) or 0)
        corner_order = int(p.get("corners_order", 0) or 0)
        fk_order = int(p.get("freekicks_order", 0) or 0)

        pen_xg_boost = 0.0
        set_piece_xa_boost = 0.0

        if pen_order == 1:
            # Small forward-looking boost only (FPL xG already includes pen xG)
            pen_xg_boost = 0.015
        elif pen_order == 2:
            pen_xg_boost = 0.005

        if corner_order == 1:
            set_piece_xa_boost += 0.03
        if fk_order == 1:
            set_piece_xa_boost += 0.02
            xg_per90 += 0.01  # direct FK goal threat

        xg_per90 += pen_xg_boost
        xa_per90 += set_piece_xa_boost

        player_gw_xpts = {}
        fix_list = upcoming.get(p["team_id"], [])

        for fix in fix_list:
            gw = fix["gw"]
            opp_team = teams_map.get(fix["opp_id"], {})
            opp_short = opp_team.get("short_name", "???")

            # Get opponent defensive strength from odds
            opp_odds = odds_by_fpl.get(opp_short, {})
            team_attack_odds = odds_by_fpl.get(team_short, {})

            # Get Elo ratings for both teams
            team_elo = elo_by_fpl.get(team_short, avg_elo)
            opp_elo = elo_by_fpl.get(opp_short, avg_elo)

            # Elo-derived strength (normalised: 1.0 = average)
            # Higher Elo = stronger team
            team_elo_str = team_elo / avg_elo if avg_elo > 0 else 1.0
            opp_elo_str = opp_elo / avg_elo if avg_elo > 0 else 1.0

            # Blend odds-based and Elo-based opponent strength
            # Elo is more dynamic (updates after every match), odds are market-informed
            opp_def_str_odds = opp_odds.get("defence_strength", 1.0)
            team_atk_str_odds = team_attack_odds.get("attack_strength", 1.0)

            # If we have Elo data, blend 50/50 with odds; otherwise use odds only
            if elo_by_fpl:
                # Elo attack proxy: team_elo_str (strong team scores more)
                # Elo defence proxy: 1/opp_elo_str (weak opponent concedes more)
                opp_def_str = (opp_def_str_odds * 0.5) + ((1.0 / max(opp_elo_str, 0.5)) * 0.5)
                team_atk_str = (team_atk_str_odds * 0.5) + (team_elo_str * 0.5)
            else:
                opp_def_str = opp_def_str_odds
                team_atk_str = team_atk_str_odds

            # Scale factor: easier opponent = higher xG
            # opp_def_str > 1 means they concede more → easier
            scale = (opp_def_str * 0.5 + team_atk_str * 0.3 + 0.2)
            home_boost = 1.1 if fix["home"] else 0.95

            adj_xg = xg_per90 * scale * home_boost
            adj_xa = xa_per90 * scale * home_boost

            # Clean sheet probability
            opp_atk_str = opp_odds.get("attack_strength", 1.0)
            team_def_str = team_attack_odds.get("defence_strength", 1.0)

            # Base CS from odds — use form-weighted xGC if available, else season
            actual_xgc_per90 = float(p.get("xgc_per90", 0) or 0)
            if form_xgc is not None and form_xgc > 0:
                # Blend form xGC with season xGC (form is more recent)
                actual_xgc_per90 = form_xgc * 0.65 + actual_xgc_per90 * 0.35
            if pos in [1, 2] and actual_xgc_per90 > 0:
                # Use actual xGC as a strong signal for defensive quality
                # Poisson approximation: P(0 goals) ≈ e^(-xGC)
                base_cs = math.exp(-actual_xgc_per90 * opp_atk_str)
            else:
                # Fallback: odds-based estimate, penalised by opponent attack
                base_cs = 0.30 * (1.0 / max(opp_atk_str, 0.5))
                # Penalise teams that concede a lot (team_def_str > 1 = concede more)
                base_cs *= (1.0 / max(team_def_str, 0.5))
                base_cs = min(base_cs, 0.50)

            # Blend with odds-derived CS if available (but don't override Poisson)
            team_cs_from_odds = team_attack_odds.get("cs_prob")
            if team_cs_from_odds and actual_xgc_per90 == 0:
                cs_prob = (base_cs * 0.4 + team_cs_from_odds * 0.6)
            else:
                cs_prob = base_cs

            # Hard cap — no team keeps a CS more than 50% of the time
            cs_prob = min(cs_prob, 0.50)

            # Apply recent form adjustment: if team has lost 3+ of last 5, reduce CS further
            try:
                team_form_list = p["team_form"] if "team_form" in p.index else []
                if isinstance(team_form_list, list) and len(team_form_list) >= 3:
                    recent_losses = sum(1 for r in team_form_list[:5] if r == "L")
                    if recent_losses >= 3:
                        cs_prob *= 0.6  # 40% penalty for bad recent form
                    elif recent_losses >= 2:
                        cs_prob *= 0.8  # 20% penalty
            except (KeyError, TypeError):
                pass

            # Calculate expected FPL points for this fixture
            xpts = 0.0

            # Appearance points (2 pts if 60+ mins, 1 pt if <60 mins)
            xpts += 2.0 * full_game_prob + 1.0 * max(play_prob - full_game_prob, 0)

            # Goal points (scale by expected 90s played, not just binary)
            xpts += adj_xg * expected_90s * PTS_GOAL.get(pos, 4)

            # Assist points
            xpts += adj_xa * expected_90s * PTS_ASSIST

            # Clean sheet points (GK and DEF mainly — need 60+ mins)
            xpts += cs_prob * PTS_CS.get(pos, 0) * full_game_prob

            # Bonus points estimate (proportional to involvement)
            xpts += PTS_BONUS_AVG * play_prob

            # Goals conceded penalty for GK/DEF
            # FPL rule: -1 point per 2 goals conceded (i.e. -0.5 per goal) from goal 1
            if pos in [1, 2]:
                expected_conceded = league_avg_goals * opp_atk_str
                if not fix["home"]:
                    expected_conceded *= 1.1  # away teams concede slightly more
                xpts -= expected_conceded * 0.5 * full_game_prob

            # Save points for GKs: ~1pt per 3 saves
            # Better estimate: GKs on bad teams face more shots but also concede more
            # Average ~3.5 saves/game across PL, but scale by opponent attack
            if pos == 1:
                expected_saves = 3.5 * opp_atk_str * 0.7  # not all shots are saveable
                save_points = (expected_saves / 3.0)  # 1 pt per 3 saves
                xpts += save_points * full_game_prob

            # ============================================================
            # DEFENSIVE CONTRIBUTION (DefCon) POINTS
            # ============================================================
            # 2025/26 rule — ALL outfield players can earn DefCon:
            #   DEFs: +2 pts for 10+ CBIT (clearances, blocks, interceptions, tackles)
            #   MIDs: +2 pts for 12+ CBIRT (CBIT + ball recoveries)
            #   FWDs: +2 pts for 12+ CBIRT (CBIT + ball recoveries)
            #   GKs: NOT eligible for DefCon
            # Capped at +2 per match.
            #
            # Key insight: DefCon is fixture-DEPENDENT but inversely to CS.
            # Facing a strong attacker = more defensive actions = higher DefCon chance
            # but lower CS chance. This makes DefCon-heavy players (Senesi, Tarkowski,
            # Caicedo, Rice) valuable across ALL fixture difficulties.
            defcon_per90 = float(p.get("defcon_per90", 0) or 0)
            if defcon_per90 > 0 and pos in [2, 3, 4] and nineties >= 3:
                # defcon_per90 from FPL API = DC points earned per 90 (0-2 scale)
                #
                # Position-specific scaling:
                # DEFs need 10 CBIT — CBs hit this regularly (30-70% of games)
                # MIDs need 12 CBIRT — only elite CDMs hit this (15-35% of games)
                # FWDs need 12 CBIRT — almost never hit this (<10% of games)
                #
                # The API's defcon_per90 already reflects actual DC points earned,
                # but we still need position-specific dampening because:
                # 1. MIDs/FWDs have a higher threshold (12 vs 10)
                # 2. Their rates are less stable/predictable
                # 3. We want to avoid over-projecting for non-defensive players

                raw_prob = defcon_per90 / 2.0  # 0-1 scale

                if pos == 2:  # DEF
                    # Conservative but fair — CBs are the primary DefCon earners
                    defcon_prob = min((raw_prob ** 0.5) * 0.6, 0.70)
                elif pos == 3:  # MID
                    # Much more conservative — only elite CDMs earn DC regularly
                    # Most MIDs (attackers, wingers, AMs) almost never hit 12 CBIRT
                    defcon_prob = min((raw_prob ** 0.5) * 0.35, 0.40)
                else:  # FWD (pos == 4)
                    # Extremely rare — almost no forwards hit 12 CBIRT
                    defcon_prob = min((raw_prob ** 0.5) * 0.15, 0.20)

                # Mild fixture adjustment
                defcon_prob *= (0.9 + 0.1 * opp_atk_str)
                defcon_prob = min(defcon_prob, 0.70 if pos == 2 else 0.40 if pos == 3 else 0.20)

                defcon_xpts = 2.0 * defcon_prob * full_game_prob
                xpts += defcon_xpts

            # Accumulate xPts — important for DGWs where a player has 2 fixtures in same GW
            gw_xpts_so_far = player_gw_xpts.get(gw, 0)
            player_gw_xpts[gw] = round(gw_xpts_so_far + max(xpts, 0), 2)

            # Store breakdown for this fixture
            if pid not in xpts_breakdown:
                xpts_breakdown[pid] = {}
            if gw not in xpts_breakdown[pid]:
                xpts_breakdown[pid][gw] = {
                    "opponent": opp_short,
                    "home": fix["home"],
                    "xg_per90": round(xg_per90, 3),
                    "xa_per90": round(xa_per90, 3),
                    "adj_xg": round(adj_xg, 3),
                    "adj_xa": round(adj_xa, 3),
                    "play_prob": round(play_prob, 2),
                    "full_game_prob": round(full_game_prob, 2),
                    "expected_90s": round(expected_90s, 2),
                    "cs_prob": round(cs_prob, 3),
                    "opp_def_str": round(opp_def_str, 3),
                    "team_atk_str": round(team_atk_str, 3),
                    "opp_atk_str": round(opp_atk_str, 3),
                    "appearance_pts": round(2.0 * full_game_prob + 1.0 * max(play_prob - full_game_prob, 0), 2),
                    "goal_pts": round(adj_xg * expected_90s * PTS_GOAL.get(pos, 4), 2),
                    "assist_pts": round(adj_xa * expected_90s * PTS_ASSIST, 2),
                    "cs_pts": round(cs_prob * PTS_CS.get(pos, 0) * full_game_prob, 2),
                    "bonus_pts": round(PTS_BONUS_AVG * play_prob, 2),
                    "conceded_pts": round(-(league_avg_goals * opp_atk_str * (1.1 if not fix["home"] else 1.0) * 0.5 * full_game_prob), 2) if pos in [1, 2] else 0,
                    "defcon_pts": round(defcon_xpts, 2) if defcon_per90 > 0 and pos in [2, 3, 4] and nineties >= 3 else 0,
                    "total": round(max(xpts, 0), 2),
                }

        # Apply blank GW override
        # Blank GW (0 fixtures) = 0 xPts — fixture loop won't have added anything,
        # but we set explicitly to be safe.
        # DGWs are already handled: the fixture loop processes both fixtures and
        # accumulates xPts, so a DGW player naturally gets ~2x a single-GW player.
        # This means DGW players only rank higher if their per-fixture xPts justify it.
        if team_fixture_counts:
            team_counts = team_fixture_counts.get(p["team_id"], {})
            for gw in list(player_gw_xpts.keys()):
                fixture_count = team_counts.get(gw, 1)
                if fixture_count == 0:
                    player_gw_xpts[gw] = 0.0

        # Total xPts over next 6 GWs
        xpts_all[pid] = player_gw_xpts

    return xpts_all, xpts_breakdown


# ============================================================
# MILP SOLVER
# ============================================================

def solve_optimal_squad(players_df, xpts_col="xpts_total", budget=1000,
                        locked_ids=None, banned_ids=None):
    """
    XI-focused MILP squad optimisation with bench cost penalty.

    locked_ids: set of player IDs that MUST be in the squad
    banned_ids: set of player IDs that MUST NOT be in the squad

    Returns: DataFrame of selected 15 players, or None
    """
    BENCH_COST_PENALTY = 0.10

    if locked_ids is None:
        locked_ids = set()
    if banned_ids is None:
        banned_ids = set()

    eligible = players_df[
        (players_df["minutes"] > 45) &
        (players_df["status"].isin(["a", "d", ""])) &
        (players_df[xpts_col] > 0)
    ].copy()

    # Also include locked players even if they fail the minutes/status filter
    if locked_ids:
        locked_players = players_df[players_df["id"].isin(locked_ids)]
        eligible = pd.concat([eligible, locked_players]).drop_duplicates(subset="id")

    # Remove banned players
    if banned_ids:
        eligible = eligible[~eligible["id"].isin(banned_ids)]

    if len(eligible) < 15:
        return None, "Not enough eligible players"

    eligible[xpts_col] = eligible[xpts_col].fillna(0).astype(float)
    eligible["now_cost"] = eligible["now_cost"].fillna(0).astype(int)
    eligible = eligible[eligible[xpts_col].notna() & eligible["now_cost"].notna()]

    if len(eligible) < 15:
        return None, "Not enough eligible players after NaN removal"

    prob = LpProblem("FPL_Squad_XI_Focused", LpMaximize)

    eligible = eligible.reset_index(drop=True)
    pid_to_idx = {row["id"]: i for i, row in eligible.iterrows()}
    xpts_vals = eligible[xpts_col].tolist()
    cost_vals = eligible["now_cost"].tolist()
    pos_vals = eligible["pos_id"].tolist()
    team_vals = eligible["team_id"].tolist()
    player_ids = eligible["id"].tolist()

    # Decision variables
    s = {pid: LpVariable(f"s_{pid}", cat="Binary") for pid in player_ids}
    xi = {pid: LpVariable(f"xi_{pid}", cat="Binary") for pid in player_ids}

    # Objective: maximise XI xPts - penalise bench cost
    # bench[pid] = s[pid] - xi[pid] (1 if on bench, 0 otherwise)
    # bench_cost_penalty = sum(bench[pid] * cost[pid] * PENALTY)
    #
    # Expanded: XI_xPts - bench_cost_penalty
    # = sum(xi * xpts) - sum((s - xi) * cost * PENALTY)
    # = sum(xi * xpts) - sum(s * cost * PENALTY) + sum(xi * cost * PENALTY)
    # = sum(xi * (xpts + cost * PENALTY)) - sum(s * cost * PENALTY)
    prob += lpSum(
        xi[pid] * (xpts_vals[pid_to_idx[pid]] + cost_vals[pid_to_idx[pid]] * BENCH_COST_PENALTY)
        - s[pid] * cost_vals[pid_to_idx[pid]] * BENCH_COST_PENALTY
        for pid in player_ids
    )

    # --- SQUAD constraints (15 players) ---
    prob += lpSum(s[pid] for pid in player_ids) == 15
    prob += lpSum(s[pid] * cost_vals[pid_to_idx[pid]] for pid in player_ids) <= budget

    for pos_id, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        pos_pids = [pid for pid in player_ids if pos_vals[pid_to_idx[pid]] == pos_id]
        prob += lpSum(s[pid] for pid in pos_pids) == count

    for team_id in set(team_vals):
        team_pids = [pid for pid in player_ids if team_vals[pid_to_idx[pid]] == team_id]
        prob += lpSum(s[pid] for pid in team_pids) <= 3

    # --- LOCKED players: must be in squad ---
    for pid in player_ids:
        if pid in locked_ids:
            prob += s[pid] == 1

    # --- XI constraints (11 from the 15) ---
    prob += lpSum(xi[pid] for pid in player_ids) == 11
    for pid in player_ids:
        prob += xi[pid] <= s[pid]

    gk_pids = [pid for pid in player_ids if pos_vals[pid_to_idx[pid]] == 1]
    def_pids = [pid for pid in player_ids if pos_vals[pid_to_idx[pid]] == 2]
    mid_pids = [pid for pid in player_ids if pos_vals[pid_to_idx[pid]] == 3]
    fwd_pids = [pid for pid in player_ids if pos_vals[pid_to_idx[pid]] == 4]

    prob += lpSum(xi[pid] for pid in gk_pids) == 1
    prob += lpSum(xi[pid] for pid in def_pids) >= 3
    prob += lpSum(xi[pid] for pid in def_pids) <= 5
    prob += lpSum(xi[pid] for pid in mid_pids) >= 2
    prob += lpSum(xi[pid] for pid in mid_pids) <= 5
    prob += lpSum(xi[pid] for pid in fwd_pids) >= 1
    prob += lpSum(xi[pid] for pid in fwd_pids) <= 3

    try:
        solver = PULP_CBC_CMD(msg=0, timeLimit=30)
        prob.solve(solver)
    except Exception as e:
        return None, f"Solver error: {e}"

    if LpStatus[prob.status] != "Optimal":
        return None, f"Solver status: {LpStatus[prob.status]}"

    selected_ids = [pid for pid in player_ids if value(s[pid]) is not None and value(s[pid]) > 0.5]
    xi_ids = [pid for pid in player_ids if value(xi[pid]) is not None and value(xi[pid]) > 0.5]
    squad = eligible[eligible["id"].isin(selected_ids)].copy()
    squad["is_xi"] = squad["id"].isin(xi_ids)

    return squad, None


def solve_best_xi(squad_df, xpts_col="xpts_next_gw"):
    """
    From a 15-man squad, pick the best starting XI using MILP.
    Must have exactly 1 GK, and valid formation (3-5-2, 4-4-2, etc.)
    """
    if squad_df is None or len(squad_df) < 11:
        return None, None

    prob = LpProblem("FPL_XI", LpMaximize)
    squad_df = squad_df.reset_index(drop=True)
    pids = squad_df["id"].tolist()
    pid_to_idx = {row["id"]: i for i, row in squad_df.iterrows()}
    xpts_vals = squad_df[xpts_col].fillna(0).tolist()
    pos_list = squad_df["pos_id"].tolist()
    x = {pid: LpVariable(f"xi_{pid}", cat="Binary") for pid in pids}

    # Objective
    prob += lpSum(x[pid] * xpts_vals[pid_to_idx[pid]] for pid in pids)

    # Exactly 11 starters
    prob += lpSum(x[pid] for pid in pids) == 11

    # Exactly 1 GK
    gk_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 1]
    prob += lpSum(x[pid] for pid in gk_pids) == 1

    # DEF: 3-5
    def_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 2]
    prob += lpSum(x[pid] for pid in def_pids) >= 3
    prob += lpSum(x[pid] for pid in def_pids) <= 5

    # MID: 2-5
    mid_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 3]
    prob += lpSum(x[pid] for pid in mid_pids) >= 2
    prob += lpSum(x[pid] for pid in mid_pids) <= 5

    # FWD: 1-3
    fwd_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 4]
    prob += lpSum(x[pid] for pid in fwd_pids) >= 1
    prob += lpSum(x[pid] for pid in fwd_pids) <= 3

    try:
        solver = PULP_CBC_CMD(msg=0, timeLimit=15)
        prob.solve(solver)
    except Exception:
        return None, None

    if LpStatus[prob.status] != "Optimal":
        return None, None

    xi_ids = [pid for pid in pids if value(x[pid]) is not None and value(x[pid]) > 0.5]
    xi = squad_df[squad_df["id"].isin(xi_ids)].copy()
    bench = squad_df[~squad_df["id"].isin(xi_ids)].copy()
    return xi, bench


# ============================================================
# DATA ENRICHMENT
# ============================================================

def enrich_data(bootstrap, fixtures, team_odds):
    """Combine all data sources into a single enriched DataFrame."""
    players_raw = bootstrap["elements"]
    teams = {t["id"]: t for t in bootstrap["teams"]}
    events = bootstrap["events"]

    # Determine the NEXT gameweek to plan for (transfers are for the future)
    # Priority: is_next (upcoming), or if is_current is not finished yet use that,
    # otherwise find the first unfinished GW
    next_gw = next((e for e in events if e.get("is_next")), None)
    current_gw_obj = next((e for e in events if e.get("is_current")), None)

    if next_gw:
        planning_gw = next_gw
    elif current_gw_obj and not current_gw_obj.get("finished"):
        planning_gw = current_gw_obj
    else:
        # All current GWs finished, find first unfinished
        unfinished = [e for e in events if not e.get("finished")]
        planning_gw = unfinished[0] if unfinished else (events[-1] if events else None)

    # For display purposes, current_gw is the latest active/completed
    current_gw = current_gw_obj or planning_gw

    # Planning GW ID — this is what we use for fixture windows and xPts
    planning_gw_id = planning_gw["id"] if planning_gw else 1
    gw_id = planning_gw_id  # all fixture lookups use this

    # Build upcoming fixtures
    upcoming = {t_id: [] for t_id in teams}
    for f in sorted(fixtures, key=lambda x: x.get("event", 0) or 0):
        ev = f.get("event")
        if ev and gw_id <= ev < gw_id + 6:
            if f["team_h"] in upcoming:
                upcoming[f["team_h"]].append({
                    "gw": ev, "opp": f["team_a"], "home": True,
                    "difficulty": f.get("team_h_difficulty", 3)
                })
            if f["team_a"] in upcoming:
                upcoming[f["team_a"]].append({
                    "gw": ev, "opp": f["team_h"], "home": False,
                    "difficulty": f.get("team_a_difficulty", 3)
                })

    # Recent results
    recent = {t_id: [] for t_id in teams}
    for f in sorted(fixtures, key=lambda x: x.get("event", 0) or 0, reverse=True):
        if f.get("finished") and f.get("team_h_score") is not None:
            h = "W" if f["team_h_score"] > f["team_a_score"] else ("D" if f["team_h_score"] == f["team_a_score"] else "L")
            a = "W" if f["team_a_score"] > f["team_h_score"] else ("D" if f["team_a_score"] == f["team_h_score"] else "L")
            if len(recent.get(f["team_h"], [])) < 5:
                recent[f["team_h"]].append(h)
            if len(recent.get(f["team_a"], [])) < 5:
                recent[f["team_a"]].append(a)

    # Build player rows
    rows = []
    for p in players_raw:
        td = teams.get(p["team"], {})
        price = p["now_cost"] / 10
        mins = p.get("minutes", 0) or 0
        pts = p.get("total_points", 0) or 0

        rows.append({
            "id": p["id"],
            "name": p.get("web_name", ""),
            "first_name": p.get("first_name", ""),
            "second_name": p.get("second_name", ""),
            "team_id": p["team"],
            "team": td.get("short_name", "???"),
            "team_name": td.get("name", "???"),
            "pos_id": p["element_type"],
            "pos": POS_MAP.get(p["element_type"], "?"),
            "price": price,
            "now_cost": p["now_cost"],
            "total_points": pts,
            "form": float(p.get("form", 0) or 0),
            "form_str": str(p.get("form", "0.0")),
            "ict_index": round(float(p.get("ict_index", 0) or 0), 1),
            "minutes": mins,
            "starts": p.get("starts", 0) or 0,
            "goals": p.get("goals_scored", 0) or 0,
            "assists": p.get("assists", 0) or 0,
            "clean_sheets": p.get("clean_sheets", 0) or 0,
            "xg_per90": float(p.get("expected_goals_per_90", 0) or 0),
            "xa_per90": float(p.get("expected_assists_per_90", 0) or 0),
            "xgi_per90": float(p.get("expected_goal_involvements_per_90", 0) or 0),
            "xgc_per90": float(p.get("expected_goals_conceded_per_90", 0) or 0),
            # Season totals for over/underperformance regression
            "xg_total": float(p.get("expected_goals", 0) or 0),
            "xa_total": float(p.get("expected_assists", 0) or 0),
            # Set piece taker status (1 = first choice, None/0 = not)
            "penalties_order": p.get("penalties_order") or 0,
            "corners_order": p.get("corners_and_indirect_freekicks_order") or 0,
            "freekicks_order": p.get("direct_freekicks_order") or 0,
            # Defensive contributions (DefCon) — new for 2025/26
            "defcon_total": int(p.get("defensive_contributions", 0) or 0),
            "defcon_per90": float(p.get("defensive_contribution_per_90", 0) or 0),
            "selected_pct": float(p.get("selected_by_percent", 0) or 0),
            "transfers_in": p.get("transfers_in_event", 0) or 0,
            "transfers_out": p.get("transfers_out_event", 0) or 0,
            "status": p.get("status", "a"),
            "chance_playing": p.get("chance_of_playing_next_round"),
            "news": p.get("news", ""),
            "ppg": float(p.get("points_per_game", 0) or 0),
            "upcoming": upcoming.get(p["team"], []),
            "team_form": recent.get(p["team"], []),
        })

    df = pd.DataFrame(rows)

    # Load form-weighted xG data from recent GWs
    player_gw_data = load_recent_gw_live_data(gw_id, n_recent=7)
    form_xg_data = compute_form_weighted_xg(player_gw_data, n_recent=7)

    # Detect blank/double gameweeks
    team_fixture_counts = detect_blank_double_gws(fixtures, gw_id, n_gws=6, teams=teams)

    # Load Club Elo ratings
    elo_ratings, elo_err = load_club_elo()

    # Build xPts model (uses planning_gw_id, so only future fixtures)
    xpts_map, xpts_breakdown = build_xpts_model(df, team_odds, teams, fixtures, gw_id,
                                 form_xg_data=form_xg_data,
                                 team_fixture_counts=team_fixture_counts,
                                 elo_ratings=elo_ratings)

    # Add xPts columns
    # xpts_next_gw = xPts for the specific next gameweek (planning_gw_id)
    df["xpts_next_gw"] = df["id"].map(
        lambda pid: xpts_map.get(pid, {}).get(planning_gw_id, 0)
    )
    # xpts_total = sum of xPts over next 6 GWs (all from planning_gw_id onwards)
    df["xpts_total"] = df["id"].map(
        lambda pid: sum(xpts_map.get(pid, {}).values())
    )

    # Avg fixture difficulty
    df["avg_difficulty"] = df["upcoming"].apply(
        lambda u: round(np.mean([f["difficulty"] for f in u[:4]]), 2) if u else 3.0
    )

    # Value metric
    df["value"] = df.apply(
        lambda r: round(r["xpts_total"] / max(r["price"], 1), 2), axis=1
    )

    return df, teams, current_gw, planning_gw_id, upcoming, fixtures, xpts_map, team_fixture_counts, xpts_breakdown


# ============================================================
# MANAGER TEAM FETCHER
# ============================================================

@st.cache_data(ttl=1800)
def fetch_manager_team(manager_id, current_gw_id):
    """
    Fetch a manager's current squad, bank, free transfers,
    and purchase prices (for correct selling price calculation).
    """
    try:
        headers = {"User-Agent": "FPL-Optimizer/2.0"}

        # 1. Basic manager info
        entry = requests.get(
            f"{FPL_BASE}/entry/{manager_id}/",
            headers=headers, timeout=15,
        ).json()

        manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}"
        team_name = entry.get("name", "Unknown")
        overall_rank = entry.get("summary_overall_rank") or "-"
        total_points = entry.get("summary_overall_points", 0)

        # 2. History endpoint — gives bank and transfer count per GW
        history = requests.get(
            f"{FPL_BASE}/entry/{manager_id}/history/",
            headers=headers, timeout=15,
        ).json()

        current_hist = history.get("current", [])
        bank = 0
        free_transfers = 1  # default

        if current_hist:
            latest_gw = current_hist[-1]
            bank = latest_gw.get("bank", 0)

            # Calculate free transfers available for NEXT gameweek
            # Logic: start with 1 FT per GW, can bank up to max 5
            # We look at all GWs to count how many were banked
            #
            # Important: the team's first GW (started_event) doesn't bank a FT
            # even if event_transfers = 0, because that's the initial squad setup
            started_event = entry.get("started_event", 1)

            ft = 1  # everyone gets 1 at start
            for gw_data in current_hist:
                gw_number = gw_data.get("event", 0)
                transfers_made = gw_data.get("event_transfers", 0)
                transfers_cost = gw_data.get("event_transfers_cost", 0)

                # Skip the GW the team was created — no FT banking on first GW
                if gw_number <= started_event:
                    ft = 1
                    continue

                if transfers_cost > 0:
                    # They took hits: used all FTs + some extra, reset to 1
                    ft = 1
                elif transfers_made == 0:
                    # Banked a FT
                    ft = min(ft + 1, 5)
                elif transfers_made <= ft:
                    # Used some/all FTs without a hit
                    remaining = ft - transfers_made
                    ft = min(remaining + 1, 5)  # +1 for the new GW's FT
                else:
                    ft = 1

            free_transfers = ft

        # 3. Transfer history — gives purchase prices (element_in_cost)
        transfers = requests.get(
            f"{FPL_BASE}/entry/{manager_id}/transfers/",
            headers=headers, timeout=15,
        ).json()

        # Build purchase price map: player_id -> purchase_price (in 0.1m units)
        # Latest transfer for each player is their purchase price
        purchase_prices = {}
        for t in transfers:
            purchase_prices[t["element_in"]] = t["element_in_cost"]

        # 4. Current picks
        picks_data = None
        for gw in [current_gw_id, current_gw_id - 1]:
            if gw < 1:
                continue
            try:
                resp = requests.get(
                    f"{FPL_BASE}/entry/{manager_id}/event/{gw}/picks/",
                    headers=headers, timeout=15,
                )
                if resp.status_code == 200:
                    picks_data = resp.json()
                    break
            except Exception:
                continue

        if picks_data is None:
            return None, "Could not fetch team picks"

        picks = picks_data.get("picks", [])
        active_chip = picks_data.get("active_chip")

        # entry_history within picks gives exact FT info in some API versions
        entry_hist = picks_data.get("entry_history", {})
        if entry_hist:
            bank = entry_hist.get("bank", bank)

        squad_ids = [p["element"] for p in picks]
        captains = {p["element"]: p.get("is_captain", False) for p in picks}
        vice_captains = {p["element"]: p.get("is_vice_captain", False) for p in picks}
        positions_in_team = {p["element"]: p.get("position", 0) for p in picks}

        # picks may contain selling_price in newer API
        selling_prices_api = {}
        for p in picks:
            if "selling_price" in p:
                selling_prices_api[p["element"]] = p["selling_price"]

        return {
            "manager_name": manager_name.strip(),
            "team_name": team_name,
            "overall_rank": overall_rank,
            "total_points": total_points,
            "bank": bank,
            "free_transfers": free_transfers,
            "squad_ids": squad_ids,
            "captains": captains,
            "vice_captains": vice_captains,
            "positions": positions_in_team,
            "active_chip": active_chip,
            "purchase_prices": purchase_prices,
            "selling_prices_api": selling_prices_api,
        }, None

    except requests.exceptions.HTTPError:
        return None, "Manager ID not found"
    except Exception as e:
        return None, str(e)


def calculate_selling_price(player_id, current_price, purchase_prices, selling_prices_api):
    """
    Calculate the correct FPL selling price.
    Rule: you get 50% of profit (rounded down).
    selling_price = purchase_price + floor((current_price - purchase_price) / 2)
    If current_price < purchase_price, selling_price = current_price (full loss).
    """
    # If the API gave us the selling price directly, use it
    if player_id in selling_prices_api:
        return selling_prices_api[player_id]

    purchase = purchase_prices.get(player_id, current_price)
    if current_price <= purchase:
        return current_price  # no profit or a loss — sell at current
    profit = current_price - purchase
    return purchase + (profit // 2)  # 50% of profit, rounded down


def find_optimal_transfers(squad_df, all_players_df, bank, free_transfers,
                           purchase_prices, selling_prices_api,
                           n_transfers=1, xpts_col="xpts_total",
                           hit_cost=4):
    """
    Find the best transfers using correct sale/buy prices and hit-aware logic.

    For each candidate transfer:
    - OUT player sold at their SELLING price (50% profit rule)
    - IN player bought at current market price (now_cost)
    - If n_transfers > free_transfers, apply -4 per extra transfer
    - Only suggest if net xPts gain > hit penalty
    """
    if squad_df is None or len(squad_df) == 0:
        return []

    squad_ids = set(squad_df["id"].tolist())

    # Available players not in squad
    available = all_players_df[
        (~all_players_df["id"].isin(squad_ids)) &
        (all_players_df["minutes"] > 45) &
        (all_players_df["status"].isin(["a", "d", ""])) &
        (all_players_df[xpts_col] > 0)
    ].copy()

    # Calculate selling price for each squad player
    squad_df = squad_df.copy()
    squad_df["sell_price"] = squad_df.apply(
        lambda r: calculate_selling_price(
            r["id"], r["now_cost"], purchase_prices, selling_prices_api
        ), axis=1
    )

    # Hit penalty: transfers beyond free ones cost 4 pts each
    extra_transfers = max(0, n_transfers - free_transfers)
    total_hit = extra_transfers * hit_cost

    suggestions = []

    if n_transfers == 1:
        for _, out_p in squad_df.iterrows():
            # Budget available = bank + selling price of outgoing player
            budget_available = bank + out_p["sell_price"]
            remaining_squad = squad_df[squad_df["id"] != out_p["id"]]
            team_counts = remaining_squad["team_id"].value_counts().to_dict()

            cands = available[
                (available["pos_id"] == out_p["pos_id"]) &
                (available["now_cost"] <= budget_available)
            ].copy()
            cands = cands[cands["team_id"].map(lambda tid: team_counts.get(tid, 0) < 3)]

            if len(cands) == 0:
                continue

            best_in = cands.loc[cands[xpts_col].idxmax()]
            xpts_gain = best_in[xpts_col] - out_p[xpts_col]
            net_gain = xpts_gain - total_hit

            if net_gain > 0.5:  # only suggest if meaningfully better
                suggestions.append({
                    "out": [out_p.to_dict()],
                    "in": [best_in.to_dict()],
                    "xpts_gain": round(xpts_gain, 1),
                    "net_gain": round(net_gain, 1),
                    "hit": total_hit,
                    "cost_change": round((best_in["now_cost"] - out_p["sell_price"]) / 10, 1),
                    "budget_after": round((budget_available - best_in["now_cost"]) / 10, 1),
                })

    elif n_transfers >= 2:
        squad_list = squad_df.to_dict("records")
        seen = set()
        for i, out1 in enumerate(squad_list):
            for j, out2 in enumerate(squad_list):
                if j <= i:
                    continue
                pair_key = (min(out1["id"], out2["id"]), max(out1["id"], out2["id"]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                freed = bank + out1["sell_price"] + out2["sell_price"]
                remaining = squad_df[~squad_df["id"].isin([out1["id"], out2["id"]])]
                tc = remaining["team_id"].value_counts().to_dict()

                best_pair_xpts = 0
                best_pair = None

                cands1 = available[
                    (available["pos_id"] == out1["pos_id"]) &
                    (available["now_cost"] <= freed)
                ]
                cands1 = cands1[cands1["team_id"].map(lambda t: tc.get(t, 0) < 3)]

                for _, in1 in cands1.nlargest(5, xpts_col).iterrows():
                    remaining_budget = freed - in1["now_cost"]
                    tc2 = tc.copy()
                    tc2[in1["team_id"]] = tc2.get(in1["team_id"], 0) + 1

                    cands2 = available[
                        (available["pos_id"] == out2["pos_id"]) &
                        (available["now_cost"] <= remaining_budget) &
                        (available["id"] != in1["id"])
                    ]
                    cands2 = cands2[cands2["team_id"].map(lambda t: tc2.get(t, 0) < 3)]

                    if len(cands2) == 0:
                        continue

                    in2 = cands2.loc[cands2[xpts_col].idxmax()]
                    pair_xpts = in1[xpts_col] + in2[xpts_col]

                    if pair_xpts > best_pair_xpts:
                        best_pair_xpts = pair_xpts
                        best_pair = (in1, in2)

                if best_pair:
                    old_xpts = out1[xpts_col] + out2[xpts_col]
                    xpts_gain = best_pair_xpts - old_xpts
                    net_gain = xpts_gain - total_hit

                    if net_gain > 0.5:
                        total_in_cost = best_pair[0]["now_cost"] + best_pair[1]["now_cost"]
                        suggestions.append({
                            "out": [out1, out2],
                            "in": [best_pair[0].to_dict(), best_pair[1].to_dict()],
                            "xpts_gain": round(xpts_gain, 1),
                            "net_gain": round(net_gain, 1),
                            "hit": total_hit,
                            "cost_change": round((total_in_cost - out1["sell_price"] - out2["sell_price"]) / 10, 1),
                            "budget_after": round((freed - total_in_cost) / 10, 1),
                        })

    suggestions.sort(key=lambda x: x["net_gain"], reverse=True)
    return suggestions[:10]




def solve_best_xi_for_gw(squad_df, xpts_map, gw_id):
    """Pick best starting XI from 15-man squad for a specific gameweek.
    Players with 0 xPts (blanking) are heavily penalised to avoid starting them."""
    if squad_df is None or len(squad_df) < 11:
        return None, None

    # Build per-GW xPts column
    sq = squad_df.copy()
    sq["xpts_gw"] = sq["id"].map(lambda pid: xpts_map.get(pid, {}).get(gw_id, 0))
    sq["xpts_gw"] = sq["xpts_gw"].fillna(0)

    # For the solver objective, penalise blanking players (xPts=0) heavily
    # so they only start if there's literally no valid formation without them
    sq["xpts_solver"] = sq["xpts_gw"].apply(lambda v: v if v > 0 else -5.0)

    prob = LpProblem(f"FPL_XI_GW{gw_id}", LpMaximize)
    sq = sq.reset_index(drop=True)
    pids = sq["id"].tolist()
    pid_to_idx = {row["id"]: i for i, row in sq.iterrows()}
    xpts_vals = sq["xpts_solver"].tolist()
    pos_list = sq["pos_id"].tolist()
    x = {pid: LpVariable(f"xi_{gw_id}_{pid}", cat="Binary") for pid in pids}

    prob += lpSum(x[pid] * xpts_vals[pid_to_idx[pid]] for pid in pids)
    prob += lpSum(x[pid] for pid in pids) == 11

    gk_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 1]
    prob += lpSum(x[pid] for pid in gk_pids) == 1
    def_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 2]
    prob += lpSum(x[pid] for pid in def_pids) >= 3
    prob += lpSum(x[pid] for pid in def_pids) <= 5
    mid_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 3]
    prob += lpSum(x[pid] for pid in mid_pids) >= 2
    prob += lpSum(x[pid] for pid in mid_pids) <= 5
    fwd_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 4]
    prob += lpSum(x[pid] for pid in fwd_pids) >= 1
    prob += lpSum(x[pid] for pid in fwd_pids) <= 3

    try:
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=10))
    except Exception:
        return None, None

    if LpStatus[prob.status] != "Optimal":
        return None, None

    xi_ids = [pid for pid in pids if value(x[pid]) is not None and value(x[pid]) > 0.5]
    xi = sq[sq["id"].isin(xi_ids)].copy()
    bench = sq[~sq["id"].isin(xi_ids)].copy()
    return xi, bench


def find_best_single_transfer_for_gw(squad_df, all_players_df, bank,
                                      purchase_prices, selling_prices_api,
                                      xpts_map, gw_id, exclude_ids=None,
                                      horizon_end=None):
    """
    Find the single best transfer considering the REMAINING HORIZON.
    
    For a GW31 transfer with horizon ending at GW35:
    - Compare players on sum(xPts from GW31 to GW35), not just GW31
    - This ensures transfers are forward-looking, not myopic
    
    Also returns the single-GW gain for display purposes.
    """
    if squad_df is None or len(squad_df) == 0:
        return None

    if exclude_ids is None:
        exclude_ids = set()
    
    # Default horizon: 6 GWs from this GW
    if horizon_end is None:
        horizon_end = gw_id + 6
    
    horizon_gws = list(range(gw_id, horizon_end))

    squad_ids = set(squad_df["id"].tolist()) | exclude_ids
    available = all_players_df[
        (~all_players_df["id"].isin(squad_ids)) &
        (all_players_df["minutes"] > 45) &
        (all_players_df["status"].isin(["a", "d", ""]))
    ].copy()

    # Remaining horizon xPts (what matters for the transfer decision)
    squad_df = squad_df.copy()
    squad_df["xpts_horizon"] = squad_df["id"].map(
        lambda pid: sum(xpts_map.get(pid, {}).get(gw, 0) for gw in horizon_gws)
    )
    available["xpts_horizon"] = available["id"].map(
        lambda pid: sum(xpts_map.get(pid, {}).get(gw, 0) for gw in horizon_gws)
    )
    
    # Also get single-GW xPts for display
    squad_df["xpts_gw"] = squad_df["id"].map(lambda pid: xpts_map.get(pid, {}).get(gw_id, 0))
    available["xpts_gw"] = available["id"].map(lambda pid: xpts_map.get(pid, {}).get(gw_id, 0))

    squad_df["sell_price"] = squad_df.apply(
        lambda r: calculate_selling_price(r["id"], r["now_cost"], purchase_prices, selling_prices_api),
        axis=1,
    )

    best = None
    best_gain = -999

    for _, out_p in squad_df.iterrows():
        budget_avail = bank + out_p["sell_price"]
        remaining = squad_df[squad_df["id"] != out_p["id"]]
        tc = remaining["team_id"].value_counts().to_dict()

        cands = available[
            (available["pos_id"] == out_p["pos_id"]) &
            (available["now_cost"] <= budget_avail)
        ]
        cands = cands[cands["team_id"].map(lambda tid: tc.get(tid, 0) < 3)]

        if len(cands) == 0:
            continue

        # Pick best by HORIZON xPts, not single GW
        top = cands.loc[cands["xpts_horizon"].idxmax()]
        horizon_gain = top["xpts_horizon"] - out_p["xpts_horizon"]
        gw_gain = top["xpts_gw"] - out_p["xpts_gw"]
        
        if horizon_gain > best_gain and horizon_gain > 0.05:
            best_gain = horizon_gain
            best = {
                "out": out_p.to_dict(),
                "in": top.to_dict(),
                "xpts_gain": round(horizon_gain, 2),  # horizon gain for decision-making
                "xpts_gw_gain": round(gw_gain, 2),     # single GW gain for display
                "new_bank": int(budget_avail - top["now_cost"]),
            }

    return best


def solve_free_hit_squad(all_players_df, xpts_map, gw_id, budget=1000, locked_ids=None):
    """Free Hit: pick best possible 15-man squad for a single GW."""
    if locked_ids is None:
        locked_ids = set()
    eligible = all_players_df[
        (all_players_df["minutes"] > 45) &
        (all_players_df["status"].isin(["a", "d", ""]))
    ].copy()
    # Include locked players even if they fail filters
    if locked_ids:
        locked_players = all_players_df[all_players_df["id"].isin(locked_ids)]
        eligible = pd.concat([eligible, locked_players]).drop_duplicates(subset="id")
    eligible["xpts_gw"] = eligible["id"].map(lambda pid: xpts_map.get(pid, {}).get(gw_id, 0))
    # For Free Hit, only pick players who actually have a fixture this GW
    # But keep locked players even if they blank
    has_fixture = eligible["xpts_gw"] > 0
    is_locked = eligible["id"].isin(locked_ids)
    eligible = eligible[has_fixture | is_locked].copy()
    eligible["xpts_gw"] = eligible["xpts_gw"].fillna(0)
    eligible["now_cost"] = eligible["now_cost"].fillna(0).astype(int)
    if len(eligible) < 15:
        return None

    eligible = eligible.reset_index(drop=True)
    pid_map = {row["id"]: i for i, row in eligible.iterrows()}
    xv = eligible["xpts_gw"].tolist()
    cv = eligible["now_cost"].tolist()
    pv = eligible["pos_id"].tolist()
    tv = eligible["team_id"].tolist()
    pids = eligible["id"].tolist()

    prob = LpProblem(f"FH_GW{gw_id}", LpMaximize)
    x = {pid: LpVariable(f"fh_{gw_id}_{pid}", cat="Binary") for pid in pids}
    prob += lpSum(x[pid] * xv[pid_map[pid]] for pid in pids)
    prob += lpSum(x[pid] * cv[pid_map[pid]] for pid in pids) <= budget
    prob += lpSum(x[pid] for pid in pids) == 15
    for pos_id, cnt in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        prob += lpSum(x[pid] for pid in pids if pv[pid_map[pid]] == pos_id) == cnt
    for tid in set(tv):
        prob += lpSum(x[pid] for pid in pids if tv[pid_map[pid]] == tid) <= 3
    # Locked players must be in squad
    for pid in pids:
        if pid in locked_ids:
            prob += x[pid] == 1
    try:
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=30))
    except Exception:
        return None
    if LpStatus[prob.status] != "Optimal":
        return None
    sel = [pid for pid in pids if value(x[pid]) is not None and value(x[pid]) > 0.5]
    return eligible[eligible["id"].isin(sel)].copy()


def solve_wildcard_squad(all_players_df, xpts_map, planning_gw, n_future, budget=1000,
                         team_fixture_counts=None, locked_ids=None):
    """
    Wildcard: best 15-man squad optimised for total xPts over remaining GWs,
    with per-GW XI awareness and locked player support.
    """
    if locked_ids is None:
        locked_ids = set()
    gw_range = list(range(planning_gw, planning_gw + n_future))

    eligible = all_players_df[
        (all_players_df["minutes"] > 45) &
        (all_players_df["status"].isin(["a", "d", ""]))
    ].copy()

    # Include locked players even if they fail filters
    if locked_ids:
        locked_players = all_players_df[all_players_df["id"].isin(locked_ids)]
        eligible = pd.concat([eligible, locked_players]).drop_duplicates(subset="id")

    # Calculate total xPts across horizon
    eligible["xpts_rem"] = eligible["id"].map(
        lambda pid: sum(xpts_map.get(pid, {}).get(gw, 0) for gw in gw_range)
    )
    eligible = eligible[eligible["xpts_rem"] > 0].copy()

    # Re-include locked players even if xpts_rem is 0 (they might blank but user wants them)
    if locked_ids:
        locked_missing = all_players_df[
            (all_players_df["id"].isin(locked_ids)) &
            (~all_players_df["id"].isin(eligible["id"]))
        ].copy()
        if len(locked_missing) > 0:
            locked_missing["xpts_rem"] = locked_missing["id"].map(
                lambda pid: sum(xpts_map.get(pid, {}).get(gw, 0) for gw in gw_range)
            )
            eligible = pd.concat([eligible, locked_missing]).drop_duplicates(subset="id")
    eligible["xpts_rem"] = eligible["xpts_rem"].fillna(0)
    eligible["now_cost"] = eligible["now_cost"].fillna(0).astype(int)
    if len(eligible) < 15:
        return None

    # Pre-compute which players have a fixture in each GW
    # (a player has a fixture if their team has >= 1 fixture that GW)
    player_has_fixture = {}  # {pid: {gw: bool}}
    for _, p in eligible.iterrows():
        pid = p["id"]
        tid = p["team_id"]
        player_has_fixture[pid] = {}
        for gw in gw_range:
            if team_fixture_counts:
                fc = team_fixture_counts.get(tid, {}).get(gw, 1)
                player_has_fixture[pid][gw] = (fc > 0)
            else:
                # No fixture count data — check if xPts > 0 as proxy
                player_has_fixture[pid][gw] = (xpts_map.get(pid, {}).get(gw, 0) > 0)

    eligible = eligible.reset_index(drop=True)
    pid_map = {row["id"]: i for i, row in eligible.iterrows()}
    xv = eligible["xpts_rem"].tolist()
    cv = eligible["now_cost"].tolist()
    pv = eligible["pos_id"].tolist()
    tv = eligible["team_id"].tolist()
    pids = eligible["id"].tolist()

    prob = LpProblem("Wildcard", LpMaximize)
    x = {pid: LpVariable(f"wc_{pid}", cat="Binary") for pid in pids}

    # Per-GW XI variables: xi[gw][pid] = 1 if player starts in that GW
    # This properly handles blanks — a blanking player is benched that GW
    BENCH_WEIGHT = 0.05
    xi_gw = {}
    for gw in gw_range:
        xi_gw[gw] = {pid: LpVariable(f"wcxi_{gw}_{pid}", cat="Binary") for pid in pids}

    # Pre-compute per-player per-GW xPts
    player_gw_xpts = {}
    for pid in pids:
        player_gw_xpts[pid] = {}
        for gw in gw_range:
            player_gw_xpts[pid][gw] = xpts_map.get(pid, {}).get(gw, 0)

    # Objective: sum over all GWs of (XI players at full value + bench at discount)
    obj_terms = []
    for gw in gw_range:
        for pid in pids:
            gw_xpts = player_gw_xpts[pid][gw]
            obj_terms.append(xi_gw[gw][pid] * gw_xpts * (1.0 - BENCH_WEIGHT))
            obj_terms.append(x[pid] * gw_xpts * BENCH_WEIGHT)
    prob += lpSum(obj_terms)

    # Budget
    prob += lpSum(x[pid] * cv[pid_map[pid]] for pid in pids) <= budget

    # Squad = 15
    prob += lpSum(x[pid] for pid in pids) == 15

    # Per-GW XI constraints
    for gw in gw_range:
        # XI = 11 per GW, subset of squad
        prob += lpSum(xi_gw[gw][pid] for pid in pids) == 11
        for pid in pids:
            prob += xi_gw[gw][pid] <= x[pid]
            # Cannot start if blanking (0 xPts)
            if player_gw_xpts[pid][gw] == 0:
                prob += xi_gw[gw][pid] == 0

        # XI formation per GW
        prob += lpSum(xi_gw[gw][pid] for pid in pids if pv[pid_map[pid]] == 1) == 1
        prob += lpSum(xi_gw[gw][pid] for pid in pids if pv[pid_map[pid]] == 2) >= 3
        prob += lpSum(xi_gw[gw][pid] for pid in pids if pv[pid_map[pid]] == 2) <= 5
        prob += lpSum(xi_gw[gw][pid] for pid in pids if pv[pid_map[pid]] == 3) >= 2
        prob += lpSum(xi_gw[gw][pid] for pid in pids if pv[pid_map[pid]] == 3) <= 5
        prob += lpSum(xi_gw[gw][pid] for pid in pids if pv[pid_map[pid]] == 4) >= 1
        prob += lpSum(xi_gw[gw][pid] for pid in pids if pv[pid_map[pid]] == 4) <= 3

    # Position constraints for squad
    for pos_id, cnt in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        prob += lpSum(x[pid] for pid in pids if pv[pid_map[pid]] == pos_id) == cnt

    # Max 3 per team
    for tid in set(tv):
        prob += lpSum(x[pid] for pid in pids if tv[pid_map[pid]] == tid) <= 3

    # Locked players must be in squad
    for pid in pids:
        if pid in locked_ids:
            prob += x[pid] == 1

    try:
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=60))
    except Exception:
        return None
    if LpStatus[prob.status] != "Optimal":
        # If infeasible (too many blanks), fall back to simpler model without per-GW XI
        prob2 = LpProblem("Wildcard_relaxed", LpMaximize)
        x2 = {pid: LpVariable(f"wc2_{pid}", cat="Binary") for pid in pids}
        prob2 += lpSum(x2[pid] * xv[pid_map[pid]] for pid in pids)
        prob2 += lpSum(x2[pid] * cv[pid_map[pid]] for pid in pids) <= budget
        prob2 += lpSum(x2[pid] for pid in pids) == 15
        for pos_id, cnt in [(1, 2), (2, 5), (3, 5), (4, 3)]:
            prob2 += lpSum(x2[pid] for pid in pids if pv[pid_map[pid]] == pos_id) == cnt
        for tid in set(tv):
            prob2 += lpSum(x2[pid] for pid in pids if tv[pid_map[pid]] == tid) <= 3
        try:
            prob2.solve(PULP_CBC_CMD(msg=0, timeLimit=45))
        except Exception:
            return None
        if LpStatus[prob2.status] != "Optimal":
            return None
        sel = [pid for pid in pids if value(x2[pid]) is not None and value(x2[pid]) > 0.5]
        return eligible[eligible["id"].isin(sel)].copy()

    sel = [pid for pid in pids if value(x[pid]) is not None and value(x[pid]) > 0.5]
    return eligible[eligible["id"].isin(sel)].copy()


def find_best_captain(squad_df, xpts_map, gw_id):
    """Find best captain (highest xPts) for a specific GW."""
    if squad_df is None or len(squad_df) == 0:
        return None
    sq = squad_df.copy()
    sq["xpts_gw"] = sq["id"].map(lambda pid: xpts_map.get(pid, {}).get(gw_id, 0))
    return sq.loc[sq["xpts_gw"].idxmax()]


def build_rolling_plan(my_squad_df, all_players_df, bank, free_transfers,
                       purchase_prices, selling_prices_api, xpts_map,
                       planning_gw_id, n_gws=6, chip_schedule=None,
                       team_fixture_counts=None, locked_ids=None, banned_ids=None):
    """
    Chip-aware rolling planner.
    chip_schedule: {gw_id: chip_name} e.g. {31: "wildcard", 33: "bench_boost"}
    locked_ids: players that must NOT be sold (transfers won't suggest selling them)
    banned_ids: players that must NOT be bought (excluded from transfer candidates)
    """
    if chip_schedule is None:
        chip_schedule = {}
    if locked_ids is None:
        locked_ids = set()
    if banned_ids is None:
        banned_ids = set()

    plan = []
    current_squad = my_squad_df.copy()
    current_bank = bank
    current_ft = free_transfers
    current_purchase = purchase_prices.copy()
    current_selling = selling_prices_api.copy()
    pre_fh_squad = None

    for i in range(n_gws):
        gw = planning_gw_id + i
        chip = chip_schedule.get(gw, None)

        has_fixtures = any(xpts_map.get(pid, {}).get(gw, 0) > 0 for pid in current_squad["id"])
        if not has_fixtures:
            break

        gw_entry = {
            "gw": gw, "chip": chip, "transfer": None, "hit": 0,
            "squad": current_squad.copy(), "xi": None, "bench": None,
            "captain": None, "captain_multiplier": 2, "bench_boost": False,
        }

        # === FREE HIT ===
        if chip == "free_hit":
            pre_fh_squad = current_squad.copy()
            total_val = int(current_bank + current_squad["now_cost"].sum())
            fh_pool = all_players_df[~all_players_df["id"].isin(banned_ids)] if banned_ids else all_players_df
            fh_squad = solve_free_hit_squad(fh_pool, xpts_map, gw, total_val, locked_ids=locked_ids)
            if fh_squad is not None:
                gw_entry["squad"] = fh_squad
                xi, bench = solve_best_xi_for_gw(fh_squad, xpts_map, gw)
            else:
                xi, bench = solve_best_xi_for_gw(current_squad, xpts_map, gw)
            gw_entry["xi"] = xi
            gw_entry["bench"] = bench
            gw_entry["captain"] = find_best_captain(xi, xpts_map, gw) if xi is not None else None
            current_ft = min(current_ft + 1, 5)
            plan.append(gw_entry)
            continue

        # Restore after free hit
        if pre_fh_squad is not None:
            current_squad = pre_fh_squad
            pre_fh_squad = None

        # === WILDCARD ===
        if chip == "wildcard":
            total_val = int(current_bank + current_squad["now_cost"].sum())
            wc_pool = all_players_df[~all_players_df["id"].isin(banned_ids)] if banned_ids else all_players_df
            wc_squad = solve_wildcard_squad(wc_pool, xpts_map, gw, n_gws - i, total_val,
                                               team_fixture_counts=team_fixture_counts,
                                               locked_ids=locked_ids)
            if wc_squad is not None:
                current_squad = wc_squad
                gw_entry["squad"] = wc_squad
                for _, p in wc_squad.iterrows():
                    current_purchase[p["id"]] = p["now_cost"]
                current_bank = total_val - wc_squad["now_cost"].sum()
            current_ft = 1
            xi, bench = solve_best_xi_for_gw(current_squad, xpts_map, gw)
            gw_entry["xi"] = xi
            gw_entry["bench"] = bench
            gw_entry["captain"] = find_best_captain(xi, xpts_map, gw) if xi is not None else None
            plan.append(gw_entry)
            continue

        # === NORMAL / TC / BB ===
        transfers_made = []
        transfers_ft_used = 0
        total_hit = 0
        recently_sold = set()

        # The horizon shrinks as we move through GWs
        horizon_end = planning_gw_id + n_gws

        # Filter out banned players from the candidate pool
        transfer_pool = all_players_df[~all_players_df["id"].isin(banned_ids)] if banned_ids else all_players_df

        # Locked players can't be sold — add them to exclude set
        transfer_exclude = set(locked_ids)

        # === FORCE LOCKED PLAYERS IN ===
        # If locked players aren't in the current squad, buy them first
        # by selling the worst player in the same position
        squad_ids = set(current_squad["id"].tolist())
        locked_to_buy = locked_ids - squad_ids
        if locked_to_buy and i == 0:  # only force buys in the first GW of the plan
            for lock_pid in locked_to_buy:
                lock_player = all_players_df[all_players_df["id"] == lock_pid]
                if len(lock_player) == 0:
                    continue
                lock_p = lock_player.iloc[0]
                lock_pos = lock_p["pos_id"]
                lock_cost = lock_p["now_cost"]

                # Find the worst player in the same position to sell (not locked)
                squad_same_pos = current_squad[
                    (current_squad["pos_id"] == lock_pos) &
                    (~current_squad["id"].isin(locked_ids))
                ].copy()
                if len(squad_same_pos) == 0:
                    continue

                # Add horizon xPts for comparison
                squad_same_pos["xpts_h"] = squad_same_pos["id"].map(
                    lambda pid: sum(xpts_map.get(pid, {}).get(g, 0) for g in range(gw, horizon_end))
                )
                worst = squad_same_pos.loc[squad_same_pos["xpts_h"].idxmin()]
                sell_price = calculate_selling_price(
                    worst["id"], worst["now_cost"],
                    current_purchase, current_selling
                )

                # Check if we can afford it
                if current_bank + sell_price >= lock_cost:
                    # Make the transfer
                    transfers_made.append({
                        "out": worst.to_dict(),
                        "in": lock_p.to_dict(),
                        "xpts_gain": 0,  # forced transfer
                        "xpts_gw_gain": 0,
                        "new_bank": int(current_bank + sell_price - lock_cost),
                    })
                    recently_sold.add(worst["id"])
                    current_squad = current_squad[current_squad["id"] != worst["id"]]
                    current_squad = pd.concat([current_squad, lock_player.iloc[:1]], ignore_index=True)
                    current_bank = int(current_bank + sell_price - lock_cost)
                    current_purchase[lock_pid] = lock_cost
                    if worst["id"] in current_selling:
                        del current_selling[worst["id"]]
                    transfers_ft_used += 1

        # Keep finding improving transfers until no more gains
        # Cap at FTs + 2 hits max (so worst case is -8, never -12 or more)
        remaining_ft = max(current_ft - transfers_ft_used, 0)
        max_transfers = min(remaining_ft + 2, 7)
        for t_num in range(max_transfers):
            transfer = find_best_single_transfer_for_gw(
                current_squad, transfer_pool, current_bank,
                current_purchase, current_selling, xpts_map, gw,
                exclude_ids=recently_sold | transfer_exclude,
                horizon_end=horizon_end,
            )

            if transfer is None:
                break  # no improving transfer found

            # Is this a free transfer or a hit?
            is_free = (t_num < current_ft)
            hit_number = t_num - current_ft + 1 if not is_free else 0  # 1st hit, 2nd hit, etc.

            # Decision thresholds:
            # Free transfers: make if ANY positive gain (it's free!)
            # Hits: escalating threshold — each additional hit needs MORE justification
            #   1st hit (-4): needs horizon gain > 6.0 (50% safety margin on the 4pt cost)
            #   2nd hit (-8 cumulative): needs horizon gain > 8.0
            #   3rd hit (-12 cumulative): needs horizon gain > 10.0 (very rarely worth it)
            # This accounts for model uncertainty and diminishing returns
            if is_free:
                if transfer["xpts_gain"] < 0.05:
                    break  # negligible gain
            else:
                hit_threshold = 4.0 + (hit_number * 2.0)  # 6.0, 8.0, 10.0, 12.0...
                if transfer["xpts_gain"] < hit_threshold:
                    break  # not enough gain to justify this hit
                total_hit += 4

            # Apply transfer
            transfers_made.append(transfer)
            out_id = transfer["out"]["id"]
            in_id = transfer["in"]["id"]
            recently_sold.add(out_id)  # don't suggest buying back
            current_squad = current_squad[current_squad["id"] != out_id]
            in_player = all_players_df[all_players_df["id"] == in_id]
            if len(in_player) > 0:
                current_squad = pd.concat([current_squad, in_player.iloc[:1]], ignore_index=True)
            current_bank = transfer["new_bank"]
            current_purchase[in_id] = transfer["in"]["now_cost"]
            if out_id in current_selling:
                del current_selling[out_id]
            if is_free:
                transfers_ft_used += 1

        # Store all transfers for this GW
        gw_entry["transfers"] = transfers_made
        gw_entry["hit"] = total_hit
        gw_entry["ft_used"] = transfers_ft_used

        # FT accounting: spent some FTs, then gain 1 for next GW
        remaining_ft = current_ft - transfers_ft_used
        current_ft = min(remaining_ft + 1, 5)

        xi, bench = solve_best_xi_for_gw(current_squad, xpts_map, gw)
        gw_entry["xi"] = xi
        gw_entry["bench"] = bench
        gw_entry["squad"] = current_squad.copy()
        gw_entry["captain"] = find_best_captain(xi, xpts_map, gw) if xi is not None else None

        if chip == "triple_captain":
            gw_entry["captain_multiplier"] = 3
        if chip == "bench_boost":
            gw_entry["bench_boost"] = True

        plan.append(gw_entry)

    return plan


def get_formation_str(xi_df):
    """Get formation string like '4-4-2' from starting XI."""
    if xi_df is None:
        return "-"
    d = len(xi_df[xi_df["pos_id"] == 2])
    m = len(xi_df[xi_df["pos_id"] == 3])
    f = len(xi_df[xi_df["pos_id"] == 4])
    return f"{d}-{m}-{f}"


def render_fdr(upcoming, teams):
    """Render fixture difficulty badges."""
    badges = []
    for f in upcoming[:5]:
        opp = teams.get(f["opp"], {}).get("short_name", "???")
        pre = "" if f["home"] else "@"
        d = f.get("difficulty", 3)
        badges.append(f'<span class="fdr-{d}">{pre}{opp}</span>')
    return " ".join(badges) if badges else "-"


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header with logo — centered
    st.markdown(
        f'<div style="text-align:center; margin-bottom:0.5rem;">'
        f'<img src="data:image/png;base64,{DATUMLY_LOGO_B64}" style="height:120px;" />'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # === Refresh button ===
    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # === Load data ===
    fetch_time = datetime.now()

    with st.spinner("Loading FPL API data..."):
        bootstrap, fixtures_raw, fpl_err = load_fpl_data()

    if fpl_err or bootstrap is None:
        st.error(f"Failed to load FPL data: {fpl_err}")
        if st.button("🔄 Retry"):
            st.cache_data.clear()
            st.rerun()
        return

    with st.spinner("Loading betting odds from football-data.co.uk..."):
        odds_df, odds_err = load_betting_odds()

    odds_status = "✅ Loaded" if odds_df is not None else f"⚠️ {odds_err or 'Unavailable'}"
    team_odds = odds_to_probabilities(odds_df, TEAM_NAME_MAP) if odds_df is not None else {}

    # Check Elo status (loaded inside enrich_data but we check here for display)
    elo_check, elo_check_err = load_club_elo()
    elo_status = f"✅ {len(elo_check)} teams" if elo_check else f"⚠️ {elo_check_err or 'Unavailable'}"

    with st.spinner("Building xPts model & enriching data..."):
        df, teams, current_gw, planning_gw_id, upcoming_map, fixtures_list, xpts_map, team_fixture_counts, xpts_breakdown = enrich_data(
            bootstrap, fixtures_raw, team_odds
        )

    # === GW Info ===
    if current_gw:
        deadline = datetime.fromisoformat(current_gw["deadline_time"].replace("Z", "+00:00"))
        status = "Completed" if current_gw.get("finished") else ("In Progress" if current_gw.get("is_current") else "Upcoming")
        bc = "badge-green" if status == "Completed" else ("badge-yellow" if status == "In Progress" else "badge-blue")
        planning_str = f"Planning for GW{planning_gw_id}" if planning_gw_id != current_gw.get("id") else ""
        st.markdown(f"""<div class="gw-bar">
            <span class="gw-num">Gameweek {current_gw['id']}</span>
            <span class="gw-deadline">Deadline: {deadline.strftime('%a %d %b, %H:%M')}</span>
            <span class="badge {bc}">{status}</span>
            {f'<span class="badge badge-blue">{planning_str}</span>' if planning_str else ''}
            <span style="color:#5a6580; font-size:0.68rem;">
                Odds: {odds_status} · Elo: {elo_status} ·
                {fetch_time.strftime('%d %b %H:%M')} (cached 1hr)
            </span>
        </div>""", unsafe_allow_html=True)

    # === Tabs ===
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 My Team", "📊 Dashboard", "👥 Player Projections",
        "⭐ Optimal Squad (MILP)", "🔄 Transfer Planner", "📅 Fixtures"
    ])

    active = df[df["minutes"] > 0].copy()
    qualified = df[df["minutes"] > 45].copy()

    # ==================== MY TEAM ====================
    with tab1:
        st.markdown(
            '<div class="section-header">🏠 My Team — Enter Your FPL ID</div>',
            unsafe_allow_html=True,
        )

        # FPL ID input
        col_id, col_btn = st.columns([3, 1])
        with col_id:
            fpl_id = st.text_input(
                "FPL Team ID",
                value=st.session_state.get("fpl_id", ""),
                placeholder="e.g. 123456",
                help="Find your ID in the URL when you view your team on the FPL website",
                key="fpl_id_input",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            load_team = st.button("Load My Team", use_container_width=True)

        if load_team and fpl_id:
            st.session_state["fpl_id"] = fpl_id

        if fpl_id and fpl_id.strip().isdigit():
            manager_id = int(fpl_id.strip())
            gw_id = current_gw["id"] if current_gw else 1

            with st.spinner("Fetching your team..."):
                team_data, team_err = fetch_manager_team(manager_id, gw_id)

            if team_err:
                st.error(f"Could not load team: {team_err}")
                st.info("Make sure your FPL ID is correct. You can find it in the URL when viewing your team on the FPL website.")
            elif team_data:
                # Manager info header
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Manager</div>
                        <div class="metric-value" style="font-size:1.1rem;">{team_data['team_name']}</div>
                        <div class="metric-sub">{team_data['manager_name']}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Overall Rank</div>
                        <div class="metric-value">{team_data['overall_rank']:,}</div>
                        <div class="metric-sub">{team_data['total_points']} pts</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    bank_display = team_data['bank'] / 10
                    ft_display = team_data.get('free_transfers', 1)
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Bank / Free Transfers</div>
                        <div class="metric-value">£{bank_display:.1f}m</div>
                        <div class="metric-sub">~{ft_display} FT available</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    squad_xpts = 0
                    my_squad = df[df["id"].isin(team_data["squad_ids"])].copy()
                    if len(my_squad) > 0:
                        squad_xpts = my_squad["xpts_total"].sum()
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Squad xPts (6GW)</div>
                        <div class="metric-value">{squad_xpts:.1f}</div>
                        <div class="metric-sub">{len(my_squad)} players loaded</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")

                if len(my_squad) > 0:
                    # Mark starters vs bench
                    my_squad["is_starter"] = my_squad["id"].map(
                        lambda pid: team_data["positions"].get(pid, 99) <= 11
                    )
                    my_squad["is_captain"] = my_squad["id"].map(
                        lambda pid: team_data["captains"].get(pid, False)
                    )
                    my_squad["is_vice"] = my_squad["id"].map(
                        lambda pid: team_data["vice_captains"].get(pid, False)
                    )

                    # Show current squad
                    st.subheader("Your Current Squad")
                    starters = my_squad[my_squad["is_starter"]].sort_values(
                        ["pos_id", "xpts_total"], ascending=[True, False]
                    )
                    bench = my_squad[~my_squad["is_starter"]].sort_values("pos_id")

                    # Pitch view of starters
                    for pid_val, plabel in [(4, "Forwards"), (3, "Midfielders"), (2, "Defenders"), (1, "Goalkeeper")]:
                        pp = starters[starters["pos_id"] == pid_val]
                        if len(pp) > 0:
                            st.markdown(f"<div class='pitch-row-label'>{plabel}</div>", unsafe_allow_html=True)
                            cols = st.columns(max(len(pp), 1))
                            for i, (_, p) in enumerate(pp.iterrows()):
                                sc = f"pitch-shirt-{p['pos'].lower()}"
                                cap_badge = " (C)" if p["is_captain"] else (" (V)" if p["is_vice"] else "")
                                with cols[i]:
                                    st.markdown(f"""<div style="text-align:center;">
                                        <div class="pitch-shirt {sc}">{p['xpts_next_gw']:.1f}</div>
                                        <div class="pitch-name">{p['name']}{cap_badge}</div>
                                        <div class="pitch-price">£{p['price']:.1f}m · {p['xpts_total']:.1f} xPts</div>
                                    </div>""", unsafe_allow_html=True)

                    if len(bench) > 0:
                        st.markdown("**Bench**")
                        bcols = st.columns(max(len(bench), 1))
                        for i, (_, p) in enumerate(bench.iterrows()):
                            with bcols[i]:
                                st.markdown(f"""<div style="text-align:center;opacity:0.6;">
                                    <div class="pitch-name">{p['name']}</div>
                                    <div class="pitch-price">{p['pos']} · £{p['price']:.1f}m · {p['xpts_total']:.1f} xPts</div>
                                </div>""", unsafe_allow_html=True)

                    st.markdown("")

                    # === CHIP & TRANSFER PLANNER ===
                    st.markdown(
                        '<div class="section-header">🗓️ Gameweek-by-Gameweek Planner '
                        '<span class="source-tag src-model">Rolling Planner</span></div>',
                        unsafe_allow_html=True,
                    )

                    ft_available = team_data.get("free_transfers", 1)
                    st.caption(f"You have **{ft_available} free transfer(s)** available.")

                    # Step 1: Chip selection
                    st.markdown("**Step 1: Set your chip schedule**")
                    gw_options = [planning_gw_id + i for i in range(6)]
                    chip_cols = st.columns(4)
                    with chip_cols[0]:
                        wc_gw = st.selectbox("🃏 Wildcard", ["None"] + gw_options, key="wc_gw")
                    with chip_cols[1]:
                        fh_gw = st.selectbox("⚡ Free Hit", ["None"] + gw_options, key="fh_gw")
                    with chip_cols[2]:
                        tc_gw = st.selectbox("👑 Triple Captain", ["None"] + gw_options, key="tc_gw")
                    with chip_cols[3]:
                        bb_gw = st.selectbox("💪 Bench Boost", ["None"] + gw_options, key="bb_gw")

                    chip_schedule = {}
                    if wc_gw != "None":
                        chip_schedule[int(wc_gw)] = "wildcard"
                    if fh_gw != "None":
                        chip_schedule[int(fh_gw)] = "free_hit"
                    if tc_gw != "None":
                        chip_schedule[int(tc_gw)] = "triple_captain"
                    if bb_gw != "None":
                        chip_schedule[int(bb_gw)] = "bench_boost"

                    # Validate
                    chip_gws_used = [g for g in [wc_gw, fh_gw, tc_gw, bb_gw] if g != "None"]
                    if len(chip_gws_used) != len(set(chip_gws_used)):
                        st.error("You can only play one chip per gameweek.")

                    # Show chip summary
                    if chip_schedule:
                        chip_labels = {"wildcard": "🃏 Wildcard", "free_hit": "⚡ Free Hit",
                                       "triple_captain": "👑 Triple Captain", "bench_boost": "💪 Bench Boost"}
                        chip_summary = " · ".join([
                            f"GW{gw}: {chip_labels.get(c, c)}" for gw, c in sorted(chip_schedule.items())
                        ])
                        st.info(f"Chip plan: {chip_summary}")
                    else:
                        st.markdown(
                            "<span style='color:#5a6580;font-size:0.8rem;'>No chips selected — planning with normal transfers only</span>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("")

                    # Step 2: Lock & Ban players
                    st.markdown("**Step 2: Lock & ban players**")
                    # Build player options from the full player pool
                    all_opts = df[df["minutes"] > 0].sort_values("xpts_total", ascending=False)
                    planner_labels = {
                        row["id"]: f"{row['name']} ({row['team']}, {row['pos']}, £{row['price']:.1f}m)"
                        for _, row in all_opts.iterrows()
                    }
                    # Separate current squad players for the lock dropdown
                    squad_ids = set(my_squad["id"].tolist())
                    squad_labels = {pid: planner_labels[pid] for pid in squad_ids if pid in planner_labels}
                    non_squad_labels = {pid: planner_labels[pid] for pid in planner_labels if pid not in squad_ids}

                    lock_col, ban_col = st.columns(2)
                    with lock_col:
                        planner_locked = st.multiselect(
                            "🔒 Lock (must be in squad)",
                            options=list(planner_labels.keys()),
                            format_func=lambda pid: planner_labels.get(pid, str(pid)),
                            placeholder="Players to always include...",
                            key="planner_lock",
                        )
                    with ban_col:
                        planner_banned = st.multiselect(
                            "🚫 Ban (don't buy these)",
                            options=list(non_squad_labels.keys()),
                            format_func=lambda pid: non_squad_labels.get(pid, str(pid)),
                            placeholder="Players to avoid...",
                            key="planner_ban",
                        )

                    planner_locked_ids = set(planner_locked)
                    planner_banned_ids = set(planner_banned)

                    st.markdown("")

                    # Step 3: Generate plan
                    st.markdown("**Step 3: Generate your optimal plan**")
                    generate = st.button("🚀 Generate 6-Gameweek Plan", use_container_width=True, type="primary")

                    if generate or st.session_state.get("plan_generated"):
                        st.session_state["plan_generated"] = True

                        with st.spinner("Building 6-gameweek rolling plan..."):
                            plan = build_rolling_plan(
                                my_squad, df,
                                bank=team_data["bank"],
                                free_transfers=ft_available,
                                purchase_prices=team_data.get("purchase_prices", {}),
                                selling_prices_api=team_data.get("selling_prices_api", {}),
                                xpts_map=xpts_map,
                                planning_gw_id=planning_gw_id,
                                n_gws=6,
                                chip_schedule=chip_schedule,
                                team_fixture_counts=team_fixture_counts,
                                locked_ids=planner_locked_ids,
                                banned_ids=planner_banned_ids,
                            )

                        st.markdown("")

                        # Plan summary
                        if plan:
                            total_xpts = 0
                            total_hits = 0
                            total_transfers = 0
                            for gw_e in plan:
                                xi = gw_e.get("xi")
                                if xi is not None and "xpts_gw" in xi.columns:
                                    cap = gw_e.get("captain")
                                    cap_mult = gw_e.get("captain_multiplier", 2)
                                    xi_pts = xi["xpts_gw"].sum()
                                    # Add captain bonus (extra x1 or x2)
                                    if cap is not None:
                                        cap_xpts = xpts_map.get(cap.get("id", 0), {}).get(gw_e["gw"], 0)
                                        xi_pts += cap_xpts * (cap_mult - 1)
                                    # Add bench if bench boost
                                    if gw_e.get("bench_boost") and gw_e.get("bench") is not None:
                                        xi_pts += gw_e["bench"]["xpts_gw"].sum() if "xpts_gw" in gw_e["bench"].columns else 0
                                    total_xpts += xi_pts
                                total_hits += gw_e.get("hit", 0)
                                total_transfers += len(gw_e.get("transfers", []))

                            sc1, sc2, sc3 = st.columns(3)
                            with sc1:
                                st.markdown(f"""<div class="metric-card">
                                    <div class="metric-label">Projected Total (6GW)</div>
                                    <div class="metric-value">{total_xpts:.0f} xPts</div>
                                    <div class="metric-sub">After captain bonus</div>
                                </div>""", unsafe_allow_html=True)
                            with sc2:
                                st.markdown(f"""<div class="metric-card">
                                    <div class="metric-label">Transfers Planned</div>
                                    <div class="metric-value">{total_transfers}</div>
                                    <div class="metric-sub">{total_hits}pts in hits</div>
                                </div>""", unsafe_allow_html=True)
                            with sc3:
                                chips_used = sum(1 for g in plan if g.get("chip"))
                                st.markdown(f"""<div class="metric-card">
                                    <div class="metric-label">Chips Used</div>
                                    <div class="metric-value">{chips_used}</div>
                                    <div class="metric-sub">of {len(chip_schedule)} planned</div>
                                </div>""", unsafe_allow_html=True)

                            st.markdown("")

                            for gw_entry in plan:
                                gw = gw_entry["gw"]
                                transfer = gw_entry["transfer"]
                                xi = gw_entry["xi"]
                                bench = gw_entry["bench"]
                                hit = gw_entry["hit"]
                                chip = gw_entry.get("chip")
                                captain = gw_entry.get("captain")
                                cap_mult = gw_entry.get("captain_multiplier", 2)
                                is_bb = gw_entry.get("bench_boost", False)

                                # Expander label with chip badge
                                chip_labels = {
                                    "wildcard": "🃏 WILDCARD",
                                    "free_hit": "⚡ FREE HIT",
                                    "triple_captain": "👑 TRIPLE CAPTAIN",
                                    "bench_boost": "💪 BENCH BOOST",
                                }
                                chip_str = f" — {chip_labels.get(chip, '')}" if chip else ""
                                with st.expander(f"**Gameweek {gw}**{chip_str}", expanded=(gw == planning_gw_id)):

                                    # Transfer / chip action
                                    if chip == "wildcard":
                                        squad_count = len(gw_entry.get("squad", []))
                                        st.markdown(
                                            f"<div class='transfer-card'>"
                                            f"<span style='color:#a78bfa;font-weight:700;'>🃏 WILDCARD ACTIVE</span>"
                                            f"<br><span style='color:#8892a8;font-size:0.72rem;'>"
                                            f"Full squad rebuilt via MILP solver ({squad_count} players) — optimised for remaining GWs</span>"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )
                                    elif chip == "free_hit":
                                        st.markdown(
                                            f"<div class='transfer-card'>"
                                            f"<span style='color:#38bdf8;font-weight:700;'>⚡ FREE HIT ACTIVE</span>"
                                            f"<br><span style='color:#8892a8;font-size:0.72rem;'>"
                                            f"Best possible squad for this single GW — reverts to your team next week</span>"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        transfers_list = gw_entry.get("transfers", [])
                                        gw_hit = gw_entry.get("hit", 0)
                                        ft_used = gw_entry.get("ft_used", 0)
                                        n_total = len(transfers_list)

                                        if n_total > 0:
                                            # Summary line
                                            hit_parts = []
                                            if ft_used > 0:
                                                hit_parts.append(f"{ft_used} free")
                                            paid = n_total - ft_used
                                            if paid > 0:
                                                hit_parts.append(f"{paid} hit (-{paid * 4}pts)")
                                            summary = " + ".join(hit_parts)
                                            total_gain = sum(t["xpts_gain"] for t in transfers_list)

                                            st.markdown(
                                                f"<span style='color:#8892a8;font-size:0.75rem;'>"
                                                f"**{n_total} transfer{'s' if n_total > 1 else ''}** ({summary}) · "
                                                f"Horizon xPts gain: <span style='color:#34d399;font-weight:600;'>+{total_gain:.1f}</span>"
                                                f"{f' · Net after hit: +{total_gain - gw_hit:.1f}' if gw_hit > 0 else ''}"
                                                f"</span>",
                                                unsafe_allow_html=True,
                                            )

                                            for t_idx, t in enumerate(transfers_list):
                                                o = t["out"]
                                                i_p = t["in"]
                                                sp = calculate_selling_price(
                                                    o["id"], o["now_cost"],
                                                    team_data.get("purchase_prices", {}),
                                                    team_data.get("selling_prices_api", {})
                                                )
                                                is_free = (t_idx < ft_used)
                                                tag = "Free" if is_free else "-4pt hit"
                                                tag_color = "#34d399" if is_free else "#f87171"
                                                gw_gain_str = f" · This GW: +{t.get('xpts_gw_gain', 0):.1f}" if "xpts_gw_gain" in t else ""

                                                st.markdown(f"""<div class="transfer-card">
                                                    <span class="transfer-out">▼ {o['name']}</span>
                                                    <span style="color:#5a6580;font-size:0.7rem;">
                                                        {o.get('pos','?')} · {o.get('team','?')} · SP £{sp/10:.1f}m
                                                    </span>
                                                    &nbsp;<span class="transfer-arrow">→</span>&nbsp;
                                                    <span class="transfer-in">▲ {i_p['name']}</span>
                                                    <span style="color:#5a6580;font-size:0.7rem;">
                                                        {i_p.get('pos','?')} · {i_p.get('team','?')} · £{i_p['now_cost']/10:.1f}m
                                                    </span>
                                                    <br>
                                                    <span style="color:#34d399;font-size:0.72rem;font-weight:600;">
                                                        +{t['xpts_gain']:.1f} xPts remaining horizon{gw_gain_str}
                                                    </span>
                                                    <span style="color:{tag_color};font-size:0.7rem;font-weight:600;"> · {tag}</span>
                                                    <span style="color:#5a6580;font-size:0.7rem;"> ·
                                                        £{t['new_bank']/10:.1f}m ITB
                                                    </span>
                                                </div>""", unsafe_allow_html=True)
                                        else:
                                            st.markdown(
                                                "<div class='transfer-card'>"
                                                "<span style='color:#8892a8;'>No transfer — banking free transfer</span>"
                                                "</div>",
                                                unsafe_allow_html=True,
                                            )
                                    # Best XI pitch view
                                    if xi is not None and len(xi) >= 11:
                                        formation = get_formation_str(xi)
                                        xi_total = xi["xpts_gw"].sum() if "xpts_gw" in xi.columns else 0
                                        st.markdown(
                                            f"<span style='color:#8892a8;font-size:0.78rem;'>"
                                            f"Best XI: {formation} · Projected {xi_total:.1f} xPts</span>",
                                            unsafe_allow_html=True,
                                        )

                                        for pid_val, plabel in [(4, "FWD"), (3, "MID"), (2, "DEF"), (1, "GK")]:
                                            pp = xi[xi["pos_id"] == pid_val]
                                            if len(pp) > 0:
                                                cols = st.columns(max(len(pp), 1))
                                                for ci, (_, p) in enumerate(pp.iterrows()):
                                                    sc = f"pitch-shirt-{POS_MAP.get(p['pos_id'],'mid').lower()}"
                                                    gw_xpts = p.get("xpts_gw", 0)
                                                    with cols[ci]:
                                                        st.markdown(f"""<div style="text-align:center;">
                                                            <div class="pitch-shirt {sc}">{gw_xpts:.1f}</div>
                                                            <div class="pitch-name">{p['name']}</div>
                                                            <div class="pitch-price">£{p['price']:.1f}m</div>
                                                        </div>""", unsafe_allow_html=True)

                                        if bench is not None and len(bench) > 0:
                                            if is_bb:
                                                bench_label = "💪 BENCH BOOST — all bench players score:"
                                            else:
                                                bench_label = "Bench:"
                                            bench_names = ", ".join([
                                                f"{r['name']} ({r.get('xpts_gw', 0):.1f})"
                                                for _, r in bench.iterrows()
                                            ])
                                            st.markdown(
                                                f"<span style='color:#5a6580;font-size:0.68rem;'>"
                                                f"{'💪 ' if is_bb else ''}{bench_label} {bench_names}</span>",
                                                unsafe_allow_html=True,
                                            )

                                    # Captain recommendation
                                    if captain is not None:
                                        cap_xpts = xpts_map.get(captain.get("id", 0), {}).get(gw, 0)
                                        cap_label = "Triple Captain" if cap_mult == 3 else "Captain"
                                        cap_emoji = "👑" if cap_mult == 3 else "©️"
                                        st.markdown(
                                            f"<span style='color:#fbbf24;font-size:0.78rem;font-weight:600;'>"
                                            f"{cap_emoji} {cap_label}: {captain.get('name', '?')} "
                                            f"({cap_xpts:.1f} xPts × {cap_mult} = {cap_xpts * cap_mult:.1f})</span>",
                                            unsafe_allow_html=True,
                                        )
                        else:
                            st.info("Could not build a rolling plan — not enough fixture data.")

                        # === EXPORT PLAN TO EXCEL ===
                        if plan and len(plan) > 0:
                            st.markdown("")
                            if st.button("📥 Export Plan to Excel", use_container_width=True):
                                import io
                                from openpyxl import Workbook
                                from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

                                wb = Workbook()

                                # --- Sheet 1: Plan Summary ---
                                ws_summary = wb.active
                                ws_summary.title = "Plan Summary"

                                header_font = Font(bold=True, color="FFFFFF", size=11, name="Arial")
                                header_fill = PatternFill("solid", fgColor="2D2D3D")
                                pink_fill = PatternFill("solid", fgColor="F02D6E")
                                green_font = Font(color="34D399", bold=True, name="Arial")
                                red_font = Font(color="F87171", bold=True, name="Arial")
                                default_font = Font(name="Arial", size=10)
                                thin_border = Border(
                                    bottom=Side(style="thin", color="3A3A4A")
                                )

                                # Headers
                                summary_headers = ["GW", "Chip", "Transfers", "Hits (pts)", "FTs Used",
                                                   "Formation", "XI xPts", "Captain", "Cap xPts", "Bench xPts"]
                                for col, h in enumerate(summary_headers, 1):
                                    cell = ws_summary.cell(row=1, column=col, value=h)
                                    cell.font = header_font
                                    cell.fill = header_fill
                                    cell.alignment = Alignment(horizontal="center")

                                for row_idx, gw_e in enumerate(plan, 2):
                                    gw = gw_e["gw"]
                                    chip = gw_e.get("chip", "—") or "—"
                                    transfers_list = gw_e.get("transfers", [])
                                    hit = gw_e.get("hit", 0)
                                    ft_used = gw_e.get("ft_used", 0)
                                    xi_data = gw_e.get("xi")
                                    bench_data = gw_e.get("bench")
                                    captain = gw_e.get("captain")
                                    cap_mult = gw_e.get("captain_multiplier", 2)

                                    formation = get_formation_str(xi_data) if xi_data is not None else "?"
                                    xi_xpts = xi_data["xpts_gw"].sum() if xi_data is not None and "xpts_gw" in xi_data.columns else 0
                                    bench_xpts = bench_data["xpts_gw"].sum() if bench_data is not None and "xpts_gw" in bench_data.columns else 0
                                    cap_name = captain.get("name", "?") if captain else "?"
                                    cap_xpts = xpts_map.get(captain.get("id", 0), {}).get(gw, 0) * cap_mult if captain else 0

                                    ws_summary.cell(row=row_idx, column=1, value=f"GW{gw}").font = default_font
                                    ws_summary.cell(row=row_idx, column=2, value=chip).font = default_font
                                    ws_summary.cell(row=row_idx, column=3, value=len(transfers_list)).font = default_font
                                    c_hit = ws_summary.cell(row=row_idx, column=4, value=f"-{hit}" if hit > 0 else "0")
                                    c_hit.font = red_font if hit > 0 else default_font
                                    ws_summary.cell(row=row_idx, column=5, value=ft_used).font = default_font
                                    ws_summary.cell(row=row_idx, column=6, value=formation).font = default_font
                                    ws_summary.cell(row=row_idx, column=7, value=round(xi_xpts, 1)).font = green_font
                                    ws_summary.cell(row=row_idx, column=8, value=cap_name).font = default_font
                                    ws_summary.cell(row=row_idx, column=9, value=round(cap_xpts, 1)).font = default_font
                                    ws_summary.cell(row=row_idx, column=10, value=round(bench_xpts, 1)).font = default_font

                                for col in range(1, 11):
                                    ws_summary.column_dimensions[chr(64 + col)].width = 14

                                # --- Sheet 2: Transfers ---
                                ws_transfers = wb.create_sheet("Transfers")
                                t_headers = ["GW", "Out", "Out Team", "Out Pos", "Out Price",
                                            "In", "In Team", "In Pos", "In Price",
                                            "Horizon Gain", "GW Gain", "Type", "Bank After"]
                                for col, h in enumerate(t_headers, 1):
                                    cell = ws_transfers.cell(row=1, column=col, value=h)
                                    cell.font = header_font
                                    cell.fill = header_fill
                                    cell.alignment = Alignment(horizontal="center")

                                t_row = 2
                                for gw_e in plan:
                                    gw = gw_e["gw"]
                                    ft_used = gw_e.get("ft_used", 0)
                                    for t_idx, t in enumerate(gw_e.get("transfers", [])):
                                        o = t["out"]
                                        i_p = t["in"]
                                        is_free = t_idx < ft_used
                                        ws_transfers.cell(row=t_row, column=1, value=f"GW{gw}").font = default_font
                                        ws_transfers.cell(row=t_row, column=2, value=o.get("name", "?")).font = red_font
                                        ws_transfers.cell(row=t_row, column=3, value=o.get("team", "?")).font = default_font
                                        ws_transfers.cell(row=t_row, column=4, value=o.get("pos", "?")).font = default_font
                                        ws_transfers.cell(row=t_row, column=5, value=round(o.get("now_cost", 0) / 10, 1)).font = default_font
                                        ws_transfers.cell(row=t_row, column=6, value=i_p.get("name", "?")).font = green_font
                                        ws_transfers.cell(row=t_row, column=7, value=i_p.get("team", "?")).font = default_font
                                        ws_transfers.cell(row=t_row, column=8, value=i_p.get("pos", "?")).font = default_font
                                        ws_transfers.cell(row=t_row, column=9, value=round(i_p.get("now_cost", 0) / 10, 1)).font = default_font
                                        ws_transfers.cell(row=t_row, column=10, value=t.get("xpts_gain", 0)).font = default_font
                                        ws_transfers.cell(row=t_row, column=11, value=t.get("xpts_gw_gain", 0)).font = default_font
                                        ws_transfers.cell(row=t_row, column=12, value="Free" if is_free else "-4pt Hit").font = default_font
                                        ws_transfers.cell(row=t_row, column=13, value=round(t.get("new_bank", 0) / 10, 1)).font = default_font
                                        t_row += 1

                                for col in range(1, 14):
                                    ws_transfers.column_dimensions[chr(64 + col) if col <= 26 else "A" + chr(64 + col - 26)].width = 14

                                # --- Sheet 3: Starting XIs per GW ---
                                ws_xi = wb.create_sheet("Starting XIs")
                                xi_headers = ["GW", "Player", "Team", "Pos", "Price", "xPts", "Captain"]
                                for col, h in enumerate(xi_headers, 1):
                                    cell = ws_xi.cell(row=1, column=col, value=h)
                                    cell.font = header_font
                                    cell.fill = header_fill
                                    cell.alignment = Alignment(horizontal="center")

                                xi_row = 2
                                for gw_e in plan:
                                    gw = gw_e["gw"]
                                    xi_data = gw_e.get("xi")
                                    captain = gw_e.get("captain")
                                    cap_id = captain.get("id") if captain else None

                                    if xi_data is not None:
                                        for _, p in xi_data.sort_values("pos_id").iterrows():
                                            is_cap = p["id"] == cap_id
                                            ws_xi.cell(row=xi_row, column=1, value=f"GW{gw}").font = default_font
                                            ws_xi.cell(row=xi_row, column=2, value=p["name"]).font = default_font
                                            ws_xi.cell(row=xi_row, column=3, value=p.get("team", "?")).font = default_font
                                            ws_xi.cell(row=xi_row, column=4, value=p.get("pos", "?")).font = default_font
                                            ws_xi.cell(row=xi_row, column=5, value=round(p.get("price", 0), 1)).font = default_font
                                            ws_xi.cell(row=xi_row, column=6, value=round(p.get("xpts_gw", 0), 1)).font = green_font
                                            ws_xi.cell(row=xi_row, column=7, value="(C)" if is_cap else "").font = Font(name="Arial", color="FFD700", bold=True) if is_cap else default_font
                                            xi_row += 1

                                for col in range(1, 8):
                                    ws_xi.column_dimensions[chr(64 + col)].width = 14

                                # Save to buffer
                                buf = io.BytesIO()
                                wb.save(buf)
                                buf.seek(0)

                                st.download_button(
                                    label="⬇️ Download Plan (.xlsx)",
                                    data=buf,
                                    file_name=f"datumly_plan_GW{planning_gw_id}-{planning_gw_id + 5}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True,
                                )

                    # Full squad table
                    st.markdown("")
                    st.subheader("Full Squad Breakdown")
                    sq_show = my_squad.sort_values(["is_starter", "pos_id", "xpts_total"],
                                                    ascending=[False, True, False])
                    # Add selling price column
                    sq_show = sq_show.copy()
                    sq_show["sell_price"] = sq_show.apply(
                        lambda r: calculate_selling_price(
                            r["id"], r["now_cost"],
                            team_data.get("purchase_prices", {}),
                            team_data.get("selling_prices_api", {})
                        ) / 10, axis=1
                    )
                    sq_display = sq_show[["name", "team", "pos", "price", "sell_price", "total_points",
                                          "form_str", "xpts_next_gw", "xpts_total", "is_starter"]].copy()
                    sq_display.columns = ["Player", "Team", "Pos", "Mkt Price", "Sell Price", "Pts",
                                          "Form", "xPts GW", "xPts 6GW", "Starter"]
                    sq_display["Starter"] = sq_display["Starter"].map({True: "XI", False: "Bench"})
                    sq_display = sq_display.reset_index(drop=True)
                    sq_display.index += 1
                    st.dataframe(sq_display, use_container_width=True)
                else:
                    st.warning("Could not match squad players to current data.")
        else:
            st.info("Enter your FPL Team ID above to get personalised transfer suggestions. "
                    "You can find it in the URL when you view your team on the FPL website "
                    "(e.g. fantasy.premierleague.com/entry/**123456**/event/1)")

    # ==================== DASHBOARD ====================
    with tab2:
        if len(active) > 0:
            c1, c2, c3, c4 = st.columns(4)
            top_xpts = qualified.loc[qualified["xpts_total"].idxmax()] if len(qualified) > 0 else None
            top_scorer = active.loc[active["total_points"].idxmax()]
            top_value = qualified.loc[qualified["value"].idxmax()] if len(qualified) > 0 else None
            top_form = active.loc[active["form"].idxmax()]

            cards = [
                (c1, "Highest xPts (6GW)", f"{top_xpts['xpts_total']:.1f}" if top_xpts is not None else "-",
                 f"{top_xpts['name']} (£{top_xpts['price']:.1f}m)" if top_xpts is not None else ""),
                (c2, "Top Scorer", str(int(top_scorer["total_points"])),
                 f"{top_scorer['name']} ({top_scorer['team']})"),
                (c3, "Best Value (xPts/£)", f"{top_value['value']:.1f}" if top_value is not None else "-",
                 f"{top_value['name']} (£{top_value['price']:.1f}m)" if top_value is not None else ""),
                (c4, "Hottest Form", f"{top_form['form']:.1f}",
                 f"{top_form['name']} ({top_form['team']})"),
            ]
            for col, label, val, sub in cards:
                with col:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{val}</div>
                        <div class="metric-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-header">Top xPts Picks — Next 6 Gameweeks <span class="source-tag src-model">xPts Model</span></div>', unsafe_allow_html=True)
        if len(qualified) > 0:
            tp = qualified.nlargest(15, "xpts_total")[
                ["name", "team", "pos", "price", "total_points", "xpts_next_gw", "xpts_total", "value"]
            ].copy()
            tp.columns = ["Player", "Team", "Pos", "Price (£m)", "Actual Pts", "xPts Next GW", "xPts 6GW", "Value"]
            tp = tp.reset_index(drop=True)
            tp.index += 1
            st.dataframe(tp, use_container_width=True, height=540)

        st.markdown("")
        st.markdown('<div class="section-header">Differentials — Under 10% Ownership</div>', unsafe_allow_html=True)
        diffs = qualified[(qualified["selected_pct"] < 10) & (qualified["xpts_total"] > 0)].nlargest(10, "xpts_total")
        if len(diffs) > 0:
            dd = diffs[["name", "team", "pos", "price", "selected_pct", "xpts_total", "value"]].copy()
            dd.columns = ["Player", "Team", "Pos", "Price", "Own%", "xPts 6GW", "Value"]
            dd = dd.reset_index(drop=True)
            dd.index += 1
            st.dataframe(dd, use_container_width=True, height=380)

    # ==================== PLAYER PROJECTIONS ====================
    with tab3:
        fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1, 1, 1, 1])
        with fc1:
            search = st.text_input("🔍 Search", "", key="ps2")
        with fc2:
            pos_f = st.selectbox("Position", ["All"] + list(POS_FULL.values()), key="pf2")
        with fc3:
            team_f = st.selectbox("Team", ["All"] + sorted(df["team_name"].unique().tolist()), key="tf2")
        with fc4:
            price_f = st.selectbox("Price", ["All", "Under £5m", "£5-7m", "£7-10m", "Over £10m"], key="prf2")
        with fc5:
            so = {"xPts (6GW)": "xpts_total", "xPts Next GW": "xpts_next_gw", "Total Pts": "total_points",
                  "Form": "form", "Value": "value", "ICT": "ict_index"}
            sort_f = st.selectbox("Sort by", list(so.keys()), key="sf2")

        fl = active.copy()
        if search:
            sl = search.lower()
            fl = fl[fl["name"].str.lower().str.contains(sl, na=False) |
                     fl["second_name"].str.lower().str.contains(sl, na=False)]
        if pos_f != "All":
            fl = fl[fl["pos_id"] == {v: k for k, v in POS_FULL.items()}[pos_f]]
        if team_f != "All":
            fl = fl[fl["team_name"] == team_f]
        if price_f == "Under £5m": fl = fl[fl["price"] < 5]
        elif price_f == "£5-7m": fl = fl[(fl["price"] >= 5) & (fl["price"] < 7)]
        elif price_f == "£7-10m": fl = fl[(fl["price"] >= 7) & (fl["price"] < 10)]
        elif price_f == "Over £10m": fl = fl[fl["price"] >= 10]

        fl = fl.sort_values(so[sort_f], ascending=False)
        sd = fl.head(80)[["name", "team", "pos", "price", "total_points", "form_str",
                           "xpts_next_gw", "xpts_total", "xg_per90", "xa_per90",
                           "selected_pct", "value"]].copy()
        sd.columns = ["Player", "Team", "Pos", "Price", "Pts", "Form",
                       "xPts GW", "xPts 6GW", "xG/90", "xA/90", "Own%", "Value"]
        sd = sd.reset_index(drop=True)
        sd.index += 1
        st.dataframe(sd, use_container_width=True, height=700)
        st.caption(f"Showing {min(80, len(fl))} of {len(fl)} players · xPts model blends FPL xG/xA + betting odds + Club Elo + DefCon")

        # === xPts BREAKDOWN INSPECTOR ===
        st.markdown("")
        st.markdown("**🔬 xPts Breakdown Inspector**")
        inspect_labels = {
            row["id"]: f"{row['name']} ({row['team']}, {row['pos']}, £{row['price']:.1f}m)"
            for _, row in active.sort_values("xpts_total", ascending=False).head(200).iterrows()
        }
        inspect_pid = st.selectbox(
            "Select a player to inspect",
            options=list(inspect_labels.keys()),
            format_func=lambda pid: inspect_labels.get(pid, str(pid)),
            key="inspect_player",
        )

        if inspect_pid and inspect_pid in xpts_breakdown:
            player_bd = xpts_breakdown[inspect_pid]
            player_row = df[df["id"] == inspect_pid].iloc[0] if len(df[df["id"] == inspect_pid]) > 0 else None

            if player_row is not None:
                st.markdown(
                    f"<span style='color:#8892a8;font-size:0.8rem;'>"
                    f"{player_row['name']} · {player_row['team']} · {player_row['pos']} · "
                    f"£{player_row['price']:.1f}m · {player_row['minutes']} mins · "
                    f"xG/90: {player_row['xg_per90']:.3f} · xA/90: {player_row['xa_per90']:.3f} · "
                    f"DefCon/90: {player_row.get('defcon_per90', 0):.2f}"
                    f"</span>",
                    unsafe_allow_html=True,
                )

            for gw in sorted(player_bd.keys()):
                bd = player_bd[gw]
                venue = "🏠 Home" if bd["home"] else "✈️ Away"
                st.markdown(
                    f"<div style='background:#1a1e2e;border-radius:8px;padding:0.7rem;margin:0.4rem 0;'>"
                    f"<span style='color:#f02d6e;font-weight:700;'>GW{gw}</span> "
                    f"<span style='color:#8892a8;'>vs {bd['opponent']} {venue}</span> "
                    f"<span style='color:#34d399;font-weight:700;font-size:1.1rem;'>"
                    f"= {bd['total']} xPts</span><br>"
                    f"<span style='color:#5a6580;font-size:0.75rem;'>"
                    f"Appearance: {bd['appearance_pts']} · "
                    f"Goals: {bd['goal_pts']} (adj xG: {bd['adj_xg']}) · "
                    f"Assists: {bd['assist_pts']} (adj xA: {bd['adj_xa']}) · "
                    f"CS: {bd['cs_pts']} (prob: {bd['cs_prob']:.0%}) · "
                    f"Bonus: {bd['bonus_pts']} · "
                    f"Conceded: {bd['conceded_pts']} · "
                    f"DefCon: {bd['defcon_pts']}<br>"
                    f"Play prob: {bd['play_prob']:.0%} · "
                    f"Full game: {bd['full_game_prob']:.0%} · "
                    f"Exp 90s: {bd['expected_90s']} · "
                    f"Opp def str: {bd['opp_def_str']} · "
                    f"Team atk str: {bd['team_atk_str']} · "
                    f"Opp atk str: {bd['opp_atk_str']}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
        elif inspect_pid:
            st.info("No breakdown available for this player (may not have enough minutes or upcoming fixtures).")

    # ==================== OPTIMAL SQUAD (MILP) ====================
    with tab4:
        st.markdown(
            '<div class="section-header">⭐ MILP-Optimised Squad '
            '<span class="source-tag src-model">PuLP Solver</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("XI-aware optimal squad: maximises starting XI xPts with bench cost penalty. "
                    "Budget saved on bench is redirected to XI upgrades. "
                    "Constraints: £100m, 2 GK / 5 DEF / 5 MID / 3 FWD, max 3 per team.")

        if len(qualified) > 0:
            # Player lock/ban controls
            all_player_options = qualified.sort_values("xpts_total", ascending=False)
            player_labels = {
                row["id"]: f"{row['name']} ({row['team']}, {row['pos']}, £{row['price']:.1f}m)"
                for _, row in all_player_options.iterrows()
            }

            col_lock, col_ban = st.columns(2)
            with col_lock:
                locked_selections = st.multiselect(
                    "🔒 Lock players (must include)",
                    options=list(player_labels.keys()),
                    format_func=lambda pid: player_labels.get(pid, str(pid)),
                    placeholder="e.g. Salah, Haaland...",
                )
            with col_ban:
                banned_selections = st.multiselect(
                    "🚫 Ban players (exclude)",
                    options=list(player_labels.keys()),
                    format_func=lambda pid: player_labels.get(pid, str(pid)),
                    placeholder="e.g. injured players, avoid...",
                )

            locked_ids = set(locked_selections)
            banned_ids = set(banned_selections)

            # Warn if too many locked
            if len(locked_ids) > 15:
                st.warning("You can't lock more than 15 players.")
                locked_ids = set()

            with st.spinner("Running MILP solver..."):
                squad, solve_err = solve_optimal_squad(
                    qualified, "xpts_total", 1000,
                    locked_ids=locked_ids, banned_ids=banned_ids,
                )

            if squad is not None and len(squad) == 15:
                # Solve best XI
                xi, bench = solve_best_xi(squad, "xpts_next_gw")

                total_cost = squad["now_cost"].sum() / 10
                total_xpts = squad["xpts_total"].sum()
                xi_xpts = xi["xpts_next_gw"].sum() if xi is not None else 0
                xi_cost = xi["now_cost"].sum() / 10 if xi is not None else 0
                bench_cost = bench["now_cost"].sum() / 10 if bench is not None else 0
                formation = get_formation_str(xi)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Cost", f"£{total_cost:.1f}m")
                c2.metric("XI Cost", f"£{xi_cost:.1f}m")
                c3.metric("Bench Cost", f"£{bench_cost:.1f}m")
                c4.metric("XI xPts (Next GW)", f"{xi_xpts:.1f}")
                c5.metric("Formation", formation)

                st.markdown("")

                # Render pitch view
                if xi is not None:
                    for pid, plabel in [(4, "Forwards"), (3, "Midfielders"), (2, "Defenders"), (1, "Goalkeeper")]:
                        pp = xi[xi["pos_id"] == pid]
                        if len(pp) > 0:
                            st.markdown(f"<div class='pitch-row-label'>{plabel}</div>", unsafe_allow_html=True)
                            cols = st.columns(max(len(pp), 1))
                            for i, (_, p) in enumerate(pp.iterrows()):
                                sc = f"pitch-shirt-{p['pos'].lower()}"
                                with cols[i]:
                                    st.markdown(f"""<div style="text-align:center;">
                                        <div class="pitch-shirt {sc}">{p['xpts_next_gw']:.1f}</div>
                                        <div class="pitch-name">{p['name']}</div>
                                        <div class="pitch-price">£{p['price']:.1f}m · {p['form_str']}</div>
                                    </div>""", unsafe_allow_html=True)

                if bench is not None and len(bench) > 0:
                    st.markdown("**Bench**")
                    bcols = st.columns(len(bench))
                    for i, (_, p) in enumerate(bench.iterrows()):
                        with bcols[i]:
                            st.markdown(f"""<div style="text-align:center;opacity:0.65;">
                                <div class="pitch-name">{p['name']}</div>
                                <div class="pitch-price">{p['pos']} · £{p['price']:.1f}m · {p['xpts_next_gw']:.1f}xPts</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown("")
                st.subheader("Full Squad Breakdown")
                sq = squad.sort_values(["pos_id", "xpts_total"], ascending=[True, False])
                sq_show = sq[["name", "team", "pos", "price", "total_points", "xpts_next_gw", "xpts_total", "value"]].copy()
                sq_show.columns = ["Player", "Team", "Pos", "Price", "Actual Pts", "xPts GW", "xPts 6GW", "Value"]
                sq_show = sq_show.reset_index(drop=True)
                sq_show.index += 1
                st.dataframe(sq_show, use_container_width=True)
            else:
                st.warning(f"Could not find optimal squad: {solve_err}")

    # ==================== TRANSFER PLANNER ====================
    with tab5:
        st.markdown('<div class="section-header">🔄 Transfer Suggestions <span class="source-tag src-model">xPts Model</span></div>', unsafe_allow_html=True)
        st.caption("Finds the highest-xPts replacements for underperforming popular players, matched by position.")

        if len(qualified) > 0:
            cands = qualified.nlargest(30, "xpts_total")
            outs = qualified[qualified["selected_pct"] > 15].nsmallest(15, "xpts_total")
            transfers, ui, uo = [], set(), set()

            for _, ip in cands.iterrows():
                for _, op in outs.iterrows():
                    if ip["id"] in ui or op["id"] in uo:
                        continue
                    if ip["pos_id"] != op["pos_id"] or ip["id"] == op["id"]:
                        continue
                    if ip["xpts_total"] <= op["xpts_total"] * 1.15:
                        continue
                    reasons = []
                    if ip["xpts_next_gw"] > op["xpts_next_gw"]:
                        reasons.append(f"+{ip['xpts_next_gw'] - op['xpts_next_gw']:.1f} xPts next GW")
                    if ip["avg_difficulty"] < op["avg_difficulty"]:
                        reasons.append("Easier fixtures")
                    if ip["form"] > op["form"]:
                        reasons.append("Better form")
                    if not reasons:
                        reasons.append(f"+{ip['xpts_total'] - op['xpts_total']:.1f} xPts over 6GW")
                    transfers.append({"out": op, "in": ip, "reasons": reasons})
                    ui.add(ip["id"])
                    uo.add(op["id"])
                    if len(transfers) >= 8:
                        break
                if len(transfers) >= 8:
                    break

            if transfers:
                for t in transfers:
                    o, i = t["out"], t["in"]
                    rs = " · ".join(t["reasons"])
                    st.markdown(f"""<div class="transfer-card">
                        <span class="transfer-out">▼ {o['name']}</span>
                        <span style="color:#5a6580;font-size:0.72rem;"> {o['pos']} · {o['team']} · £{o['price']:.1f}m · {o['xpts_total']:.1f}xPts</span>
                        &nbsp;<span class="transfer-arrow">→</span>&nbsp;
                        <span class="transfer-in">▲ {i['name']}</span>
                        <span style="color:#5a6580;font-size:0.72rem;"> {i['pos']} · {i['team']} · £{i['price']:.1f}m · {i['xpts_total']:.1f}xPts</span>
                        <br><span style="color:#8892a8;font-size:0.7rem;">{rs}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No clear transfer improvements found.")

        st.markdown("")
        cr, cf = st.columns(2)
        with cr:
            st.subheader("📈 Most Transferred In")
            ri = active.nlargest(10, "transfers_in")[["name", "team", "pos", "price", "transfers_in", "xpts_total"]].copy()
            ri.columns = ["Player", "Team", "Pos", "Price", "In", "xPts 6GW"]
            ri = ri.reset_index(drop=True); ri.index += 1
            st.dataframe(ri, use_container_width=True)
        with cf:
            st.subheader("📉 Most Transferred Out")
            fo = active.nlargest(10, "transfers_out")[["name", "team", "pos", "price", "transfers_out", "xpts_total"]].copy()
            fo.columns = ["Player", "Team", "Pos", "Price", "Out", "xPts 6GW"]
            fo = fo.reset_index(drop=True); fo.index += 1
            st.dataframe(fo, use_container_width=True)

    # ==================== FIXTURES ====================
    with tab6:
        st.markdown('<div class="section-header">Fixture Difficulty — Next 6 Gameweeks <span class="source-tag src-odds">Odds-enhanced</span></div>', unsafe_allow_html=True)
        gw_range = list(range(planning_gw_id, planning_gw_id + 6))

        fm = {t_id: {} for t_id in teams}
        for f in fixtures_list:
            if f.get("event") in gw_range:
                if f["team_h"] in fm:
                    fm[f["team_h"]][f["event"]] = {"opp": f["team_a"], "home": True, "diff": f.get("team_h_difficulty", 3)}
                if f["team_a"] in fm:
                    fm[f["team_a"]][f["event"]] = {"opp": f["team_h"], "home": False, "diff": f.get("team_a_difficulty", 3)}

        rows = []
        for t_id, td in teams.items():
            row = {"Team": td["short_name"]}
            diffs = []
            t_short = td["short_name"]
            t_odds = {v: team_odds.get(k) for k, v in TEAM_NAME_MAP.items()}.get(t_short, {})

            for gw in gw_range:
                fix = fm.get(t_id, {}).get(gw)
                fc = team_fixture_counts.get(t_id, {}).get(gw, 1)
                if fc == 0:
                    row[f"GW{gw}"] = "BLANK"
                elif fix:
                    opp = teams.get(fix["opp"], {}).get("short_name", "???")
                    pre = "" if fix["home"] else "@"
                    dgw = " [DGW]" if fc >= 2 else ""
                    row[f"GW{gw}"] = f"{pre}{opp} ({fix['diff']}){dgw}"
                    diffs.append(fix["diff"])
                else:
                    row[f"GW{gw}"] = "-"

            row["Avg FDR"] = round(np.mean(diffs), 1) if diffs else 3.0

            # Add odds-derived CS probability if available
            if t_odds and isinstance(t_odds, dict):
                row["CS%"] = f"{t_odds.get('cs_prob', 0)*100:.0f}%"
                row["Atk Str"] = f"{t_odds.get('attack_strength', 1):.2f}"
            else:
                row["CS%"] = "-"
                row["Atk Str"] = "-"

            rows.append(row)

        fdf = pd.DataFrame(rows).sort_values("Avg FDR").reset_index(drop=True)
        fdf.index += 1
        st.dataframe(fdf, use_container_width=True, height=740)

    # === Footer ===
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:#5a6580; font-size:0.7rem;'>"
        f"Datumly · Data-driven FPL intelligence · "
        f"FPL API + football-data.co.uk + PuLP MILP · "
        f"{datetime.now().strftime('%d %b %Y, %H:%M')}"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
