# How to make a release

1. Change the version number from `0.0.X dev` to `0.0.X` in [mne-nirs/mne_nirs/_version.py](https://github.com/mne-tools/mne-nirs/blob/master/mne_nirs/_version.py#L1)
1. Push change and wait for PR checks to go green.
1. Add new version to [mne-nirs/doc/conf.py](https://github.com/mne-tools/mne-nirs/blob/714dc6f75ebc561e7974ba7d3256fe0ae8d35174/doc/conf.py#L131) by changing `html_context, versions_dropdown`.
   Remove the stable tag from previous version and add this version as `'v0.0.X': 'v0.0.X (stable)',`
1. Push change and wait for PR checks to go green.
1. Merge PR and wait for checks to go green.
1. Pull changes locally
1. Create release locally by
   1. `pip install twine`
   1. `rm -rf dist`
   1. `python setup.py sdist`
   1. `twine upload dist/*`
1. Create a release in GitHub interface which also creates a git tag
1. Change the version number from `0.0.X` to `0.0.X+1 dev` in [mne-nirs/mne_nirs/_version.py](https://github.com/mne-tools/mne-nirs/blob/master/mne_nirs/_version.py#L1)
1. Set docs to build from 2 onwards in [conf.py](https://github.com/mne-tools/mne-nirs/blob/714dc6f75ebc561e7974ba7d3256fe0ae8d35174/doc/conf.py#L57)
1. Commit and rebuild all docs
1. Set docs to build from current version onwards in [conf.py](https://github.com/mne-tools/mne-nirs/blob/714dc6f75ebc561e7974ba7d3256fe0ae8d35174/doc/conf.py#L57)
1. Commit

Done!
