<!-- ## Listen to vocal extraction -->
## Below are the examples of vocals source separation produced by 4 models.


 - [Code](https://github.com/taras-svystun/Band-Split-RNN)
 <!-- - [Demo](https://huggingface.co/spaces/PolyAI/pheme) -->
 <!-- - [Paper](https://arxiv.org/pdf/2401.02839.pdf) -->




| Song name           | Original sample                                                                                       | Init model                                                                                       | My trained model                                                                               |   Demucs                                                                                        | MDX                                                                                               |
|---------------------|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Roots, Rock, Reggae | <audio src="s/sample_Roots_Rock_Reggae_15_sec.wav" type="audio/wav" controls preload></audio>        | <audio src="s/Amantur_model_vocals_Roots_Rock_Reggae.wav" type="audio/wav" controls preload></audio> | <audio src="s/my_model_vocals_Roots_Rock_Reggae.wav" type="audio/wav" controls preload></audio> | <audio src="s/demucs_model_vocals_Roots_Rock_Reggae.mp3" type="audio/wav" controls preload></audio> | <audio src="s/mdx_model_vocals_Roots_Rock_Reggae.mp3" type="audio/wav" controls preload></audio> |
| Hi de Hi, Hi de Ho  | <audio src="s/sample_Kool_and_the_gang_28_sec.wav" type="audio/wav" controls preload></audio>         | <audio src="s/Amantur_model_vocals_Kool_and_the_gang.wav" type="audio/wav" controls preload></audio>  | <audio src="s/my_model_vocals_Kool_and_the_gang.wav" type="audio/wav" controls preload></audio> | <audio src="s/demucs_model_vocals_Kool_and_the_gang.mp3" type="audio/wav" controls preload></audio>  | <audio src="s/mdx_model_vocals_Kool_and_the_gang.mp3" type="audio/wav" controls preload></audio> |
| 420 Louis            | <audio src="s/sample_420_Louis_14_sec.wav" type="audio/wav" controls preload></audio>                 | <audio src="s/Amantur_model_vocals_420_Louis.wav" type="audio/wav" controls preload></audio>           | <audio src="s/my_model_vocals_420_Louis.wav" type="audio/wav" controls preload></audio>         | <audio src="s/demucs_model_vocals_420_Louis.mp3" type="audio/wav" controls preload></audio>           | <audio src="s/mdx_model_vocals_420_Louis.mp3" type="audio/wav" controls preload></audio>         |


## Models summary

| Model              | SDR, dB  |
| ------------------ | ------- |
| Init               | 6.88    |
| Trained            | 8.07    |
| MDX                | 9.0     |
| Demucs             | 9.37    |

