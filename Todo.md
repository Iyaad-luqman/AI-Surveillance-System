Using https://github.com/mlfoundations/open_clip process video files for violence detection or car fire or whatever. 


Frame Skipping: Instead of processing every frame in the video, you can process every nth frame. This will significantly reduce the amount of data that needs to be processed, but it may miss important frames.

Parallel Processing: Use multiple cores or multiple machines to process the video in parallel. This can significantly reduce the processing time, but it requires more resources and more complex code.

The @torch.no_grad() decorator is used to disable gradient calculations during the forward pass, which can save memory when you only need to do a forward pass (like during inference).

VERCEL IS FREE ? ? ? ? ? ? ? ? ? ?  ?? ? ? ? ? ? ? ?

[ ] Try to get the app working open_clip
	
	It basically uses transformers and then goes to settings.yaml, fetches the classification from there, tries to get the threshold and then matches with one and then alerts the prediction.

[ ] Try support for video from the jupyter notebook code,
[ ] Add support where the user gives the context its looking for and only try to detect form it.
[ ] Try to get the timeframe of it and take a screenshot and save it. 
[ ] NLP process using nlp_interpreter and take the user_input and parse it out for violence classification
[ ] Create a Web app where there is option for selecting which process_engine to run it on.
	
	- Maybe try JS Framworks with Beautiful default compnents and tell the the AI tool to write the code.

[ ] Add a UI Where it shows the screenshot along with timeframes.
[ ] For large video files, tell the user when in IST the video start playing, and then the user can ask queries like car crash between 7pm and 8pm, so that it trims those videos specifically and then analyses that part of the code alone 




[ ] Try all the optimization techniques 