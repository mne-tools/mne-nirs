ARG mne_python_v=main
ARG nilearn_v=main
ARG mne_bids_v=main
ARG mne_nirs_v=main

FROM ghcr.io/mne-tools/mne-python-plot:${mne_python_v}

ARG nilearn_v
ARG mne_bids_v
ARG mne_nirs_v

USER root

RUN pip install statsmodels dabest fooof h5io
RUN pip install https://github.com/nilearn/nilearn/archive/${nilearn_v}.zip
RUN pip install https://github.com/mne-tools/mne-bids/archive/${mne_bids_v}.zip
RUN pip install https://github.com/mne-tools/mne-nirs/archive/${mne_nirs_v}.zip

# Copy examples
COPY ./ /home/mne_user/

USER $MNE_USER
WORKDIR $HOME_DIR

ENTRYPOINT ["tini", "-g", "--", "/usr/bin/prepare.sh"]
