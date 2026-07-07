# Metamon setup (thesis data pipeline)

[Metamon](https://github.com/UT-Austin-RPL/metamon) provides reconstructed first-person trajectories from human Showdown replays. That matches the thesis data source.

## Two Python environments (recommended)

Metamon pins its own `poke-env` fork. Your thesis scripts use `poke-env==0.15.0`. Use two venvs to avoid conflicts.

**Thesis venv** (baseline, BC, continuous_train)

```powershell
cd "C:\Users\ethan\Masters Thesis"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-thesis.txt
```

**Metamon venv**

```powershell
python -m venv .venv-metamon
.\.venv-metamon\Scripts\Activate.ps1
pip install -r requirements-metamon.txt
```

Repo clone location

```
pokemon-thesis-tools/metamon/
```

## Cache directory

Set a folder with plenty of free space (parsed replays are large).

```powershell
$env:METAMON_CACHE_DIR = "C:\Users\ethan\Masters Thesis\data\metamon"
```

Or add that line to your PowerShell profile.

## Download human replays

Start with **gen9ou** only unless you need older generations.

```powershell
.\.venv-metamon\Scripts\Activate.ps1
python scripts/setup_metamon.py --download gen9ou
```

If a previous download failed on Windows, try `--fresh`. If delete fails, use a new cache folder instead of fighting the old one.

```powershell
python scripts/setup_metamon.py --download gen9ou --cache-dir data/metamon_clean
```

Or wipe with `--fresh` after closing other Python terminals.

```powershell
python scripts/setup_metamon.py --download gen9ou --fresh
```

Check versions without downloading

```powershell
python scripts/setup_metamon.py --check
```

## Explore one battle

After gen9ou data is present

```powershell
python scripts/explore_metamon.py --format gen9ou
```

## Link to your 13-token encoder

Metamon exposes `UniversalState` and `DefaultObservationSpace`. The next integration step is a converter from Metamon states to your 13-token layout in `src/pokemon_thesis/metamon_bridge/`.

Random-bot logs in `logs/` are for pipeline testing. Thesis training should move to Metamon human replays.

## Showdown server

You can keep using your existing local server under `pokemon-thesis-tools/pokemon-showdown`. Metamon also ships a supported copy under `pokemon-thesis-tools/metamon/server/pokemon-showdown`.

## References

- [Metamon GitHub](https://github.com/UT-Austin-RPL/metamon)
- [Parsed replays on Hugging Face](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)
- [Paper / site](https://metamon.tech/)
