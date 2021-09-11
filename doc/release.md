# How to make a release

1. Change version number and doc dropdown options. E.g. [46104c8](https://github.com/mne-tools/mne-nirs/pull/295/commits/46104c8cc5f971b1cce772626869dd96993b2bb7)
    1. Change the version number from `0.0.X dev` to `0.0.X` in [mne-nirs/mne_nirs/_version.py](https://github.com/mne-tools/mne-nirs/blob/master/mne_nirs/_version.py#L1)
    2. Add new version to [mne-nirs/doc/conf.py](https://github.com/mne-tools/mne-nirs/blob/714dc6f75ebc561e7974ba7d3256fe0ae8d35174/doc/conf.py#L131) by changing `html_context, versions_dropdown`.
       Remove the stable tag from previous version and add this version as `'v0.0.X': 'v0.0.X (stable)',`
    3. Modify the changelog.md and rename the `-dev` from most recent changes
1. Push change and wait for PR checks to go green.
1. Merge PR and wait for checks to go green.
1. Clone main branch locally. `git clone git@github.com:mne-tools/mne-nirs.git`
3. Create release locally by
   1. `pip install twine`
   1. `rm -rf dist`
   1. `python setup.py sdist`
   1. `twine upload dist/*`
4. Create a release in GitHub interface which also creates a git tag
5. Bump version to dev naming and regenerate docs (e.g. [6393b6dfc6](https://github.com/mne-tools/mne-nirs/pull/321/commits/6393b6dfc6f4fb8c5068c2ec728dfecd41c11897)).
    1. Change the version number from `0.0.X` to `0.0.X+1 dev` in [mne-nirs/mne_nirs/_version.py](https://github.com/mne-tools/mne-nirs/blob/master/mne_nirs/_version.py#L1)
     1. Set docs to build all versions in [conf.py](https://github.com/mne-tools/mne-nirs/blob/714dc6f75ebc561e7974ba7d3256fe0ae8d35174/doc/conf.py#L57) by setting `smv_tag_whitelist = r'^v\d+\.\d+.\d+$'`
6. Commit which rebuild all docs
7. Set docs to build current version only in [conf.py](https://github.com/mne-tools/mne-nirs/blob/714dc6f75ebc561e7974ba7d3256fe0ae8d35174/doc/conf.py#L57) by setting `smv_tag_whitelist = None`
8. Commit

Done!
