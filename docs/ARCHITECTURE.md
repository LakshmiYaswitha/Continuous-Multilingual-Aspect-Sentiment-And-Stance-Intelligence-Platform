# Architecture

Input -> ingestion -> language detection -> preprocessing -> reference/advanced NLP models -> aspect/opinion -> ABSA -> stance -> valence/arousal -> evidence -> result engine -> dashboard/export.

The reference sentiment model is trained from user-selected Kaggle labelled data. The system does not fabricate labels for valence/arousal or ABSA training: when dedicated labelled corpora are available, their models can be added behind the same interfaces.
