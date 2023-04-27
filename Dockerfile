ARG mne_python_v=main
ARG nilearn_v=main
ARG mne_bids_v=main
ARG mne_nirs_v=main

FROM ghcr.io/mne-tools/mne-python-plot:${mne_python_v}

ARG nilearn_v
ARG mne_bids_v
ARG mne_nirs_v

USER root

RUN apt-get update && apt-get install -y --no-install-recommends git
RUN pip install statsmodels dabest fooof h5io
RUN pip install git+https://github.com/nilearn/nilearn.git@${nilearn_v}
RUN pip install git+https://github.com/mne-tools/mne-bids.git@${mne_bids_v}
RUN pip install git+https://github.com/mne-tools/mne-nirs.git@${mne_nirs_v}

# Copy examples
COPY ./ /home/mne_user/

USER $MNE_USER
WORKDIR $HOME_DIR

ENTRYPOINT ["tini", "-g", "--", "/usr/bin/prepare.sh"]
