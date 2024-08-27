ARG mne_python_v=latest
ARG nilearn_v=main
ARG mne_nirs_v=main

FROM ghcr.io/mne-tools/mne-python-plot:${mne_python_v}

ARG nilearn_v
ARG mne_nirs_v

USER mne_user
WORKDIR $HOME_DIR

RUN pip install git+https://github.com/nilearn/nilearn.git@${nilearn_v}
RUN pip install git+https://github.com/mne-tools/mne-nirs.git@${mne_nirs_v}

# Copy examples
COPY ./ /home/mne_user/

ENTRYPOINT ["tini", "-g", "--", "/usr/bin/prepare.sh"]
