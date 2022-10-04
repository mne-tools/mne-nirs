"""
.. _tut-migration-nirstoolbox:

Migrating from nirs-toolbox to MNE
==================================

This tutorial will demonstrate how to do convert a nirs-toolbox script to MNE
and highlight differences and common issues you may encounter.

nirs-toolbox is a MATLAB based software analysis
package. https://github.com/huppertt/nirs-toolbox


Basic nirs-toolbox script to be reproduced in MNE
=================================================

Below is a common example analysis performed in nirs-toolbox.
The NIRx data is loaded and trigger names are modified.
Then the intensity signal is convert to optical density then
converted to haemoglobin concentration.
After which, a GLM analysis is performed with iteratively reweighted
autoregressive noise model.
Finally a mixed effects model is run on the group level dat.


.. code-block:: matlab

   # Load data
   raw = nirs.io.loadDirectory()

   # Preprocess data
   pipeline_preprocess = nirs.modules.RenameStims(pipeline_preprocess);
   pipeline_preprocess.listOfChanges = {
    'channel_1', 'Control';
    'channel_2', 'Condition1';
    'channel_3', 'Condition2'};
   pipeline_preprocess = nirs.modules.DiscardStims(pipeline_preprocess);
   raw = pipeline_preprocess.run(raw);

   # Analyse data
   pipeline = nirs.modules.Resample();                     pipeline.Fs = 3;
   pipeline = nirs.modules.OpticalDensity(pipeline);
   pipeline = nirs.modules.BeerLambertLaw(pipeline);
   pipeline = nirs.modules.AR_IRLS(pipeline);
   subj_stats = pipeline.run(raw);

   # Group level analysis
   pipeline = nirs.modules.MixedEffects();
   pipeline.formula = 'beta ~ -1 + cond + (1|Name)';
   group_stats = pipeline.run(subj_stats);

.. contents:: Page contents
   :local:
   :depth: 2

"""

# %%
# MNE equivalent of nirs-toolbox script
# =====================================
#
# The MNE equivalent of this analysis is well described
# in :ref:`the group level tutorial <tut-fnirs-group>`.
#
# So this page will focus on the differences between the two approaches.


# %%
# Differences between toolboxes
# ---------------------------------------------
#
# MNE does not higher order autoregressive noise models
# in the GLM analysis. See https://github.com/mne-tools/mne-nirs/issues/233
#
# MNE does not have iterative reweighting of the noise estimate
# in the GLM analysis.


# %%
# Further analysis details
# ------------------------------------
#
# If there is useful processing in the nirs-toolbox
# that is not available in MNE
# please let us know by creating an issue at
# https://github.com/mne-tools/mne-nirs/issues
