# Text Summarization of Hindi News Article
We compare and evaluate the performance of the following sequence to seuqence models on hindi text:
* Seq2Seq (RNN baseline)
* Seq2Seq + Attention
* Seq2Seq + fasttext embedding
* Seq2Seq + Attention + fasttext embedding
* BART_T5

The details of the files and the folders are as follows:
* NLP_project_hindi_text_summarization.ipynb : Code for baseline implementation and attention visualization 
*  NLP_BARTT5.ipynb : Code for transfer learning and BART_T5 implementation
*  attention_visualizations: Folder containing attention visualization for few samples
*  models: Folder containing pretrained models that can be directly used for evaluation
