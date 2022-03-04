(function() {
    // adapted 2022-03 from https://mne.tools/versionwarning.js
    // adapted 2020-05 from https://scikit-learn.org/versionwarning.js
    if (location.hostname === 'mne.tools') {
        const urlParts = location.pathname.split('/');
        const version = urlParts[2];
        // see if filePath exists in the stable version of the docs
        var filePath = urlParts.slice(3).join('/');
        $.ajax({
            type: 'HEAD',
            url: `https://mne.tools/mne-nirs/stable/${filePath}`,
            error: function() {
                filePath = '';
            },
            complete: function() {
                if (version !== 'stable') {
                    // parse version to figure out which website theme classes to use
                    var pre = '<div class="container-fluid alert-danger devbar"><div class="row no-gutters"><div class="col-12 text-center">';
                    var post = '</div></div></div>';
                    var anchor = 'class="btn btn-danger font-weight-bold ml-3 my-3 align-baseline"';
                    // triage message
                    var verText = `an <strong>old version (${version})</strong>`;
                    if (version == 'dev') {
                        verText = 'the <strong>unstable development version</strong>';
                    }
                    $('body').prepend(`${pre}This is documentation for ${verText} of MNE-NIRS. <a ${anchor} href="https://mne.tools/mne-nirs/stable/${filePath}">Switch to stable version</a>${post}`);
                }
            }
        });
    }
})()
