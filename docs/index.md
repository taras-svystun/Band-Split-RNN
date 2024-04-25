## Listen to vocal extraction

 - [Code](https://github.com/PolyAI-LDN/pheme)
 - [Demo](https://huggingface.co/spaces/PolyAI/pheme)
 - [Paper](https://arxiv.org/pdf/2401.02839.pdf)



|Song name|Original sample| Init model | My trained model | ht-demucs | mdx |
|:--|--|--|--|--|--|
Roots, Rock, Reggae|<audio src="separation_examples/sample_Roots_Rock_Reggae_15_sec.wav" type="audio/wav" controls preload></audio>|
<audio src="separation_examples/Amantur_model_vocals_Roots_Rock_Reggae.wav" type="audio/wav" controls preload></audio>|
<audio src="separation_examples/my_model_vocals_Roots_Rock_Reggae.wav" type="audio/wav" controls preload></audio>|
<audio src="separation_examples/demucs_model_vocals_Roots_Rock_Reggae.mp3" type="audio/wav" controls preload></audio>|
<audio src="separation_examples/mdx_model_vocals_Roots_Rock_Reggae.mp3" type="audio/wav" controls preload></audio>|




### Inference speed with Triton-LLM (RTFs, lower is better) for short and long sentences

| Model              | *short*   | *long*    | GPU    |
| ------------------ | --------- | --------- |--------- |
| MQTTS (100M)       | 1.930     | 1.842     | A100  |
| PHEME-SMALL (100M) | **0.133** | **0.133** | A100   |
| PHEME-LARGE (300M) | 0.143     | 0.143     | A100     |

