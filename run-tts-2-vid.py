import glob
import os
from time import sleep
from tqdm import tqdm
import moviepy as mpy

#2025-07-26
post_urls = [
 "https://www.reddit.com/r/AskReddit/comments/1i88o15/if_someone_grabbed_you_out_of_your_chair_right/",
"https://www.reddit.com/r/AskReddit/comments/1hzglw3/for_those_who_used_a_computer_between_1995_and/",
"https://www.reddit.com/r/AskReddit/comments/1i07e15/what_was_the_biggest_waste_of_money_in_human/",
"https://www.reddit.com/r/AskReddit/comments/1i1czin/whats_a_dead_giveaway_someone_grew_up_as_an_only/",
"https://www.reddit.com/r/AskReddit/comments/1hv05cp/what_is_a_rich_person_thing_that_you_would_be/",
"https://www.reddit.com/r/AskReddit/comments/1i5ei0h/what_ages_a_person_really_quickly/",
    "https://www.reddit.com/r/AskReddit/comments/1hz5e1y/for_those_earning_over_10k_per_month_what_do_you/",
    "https://www.reddit.com/r/AskReddit/comments/1hwhvop/in_25_years_when_someone_asks_what_life_was_like/",
    "https://www.reddit.com/r/AskReddit/comments/1i8rjjd/what_is_a_product_that_you_swore_by_your_whole/",
"https://www.reddit.com/r/AskReddit/comments/1ht0jmq/whos_your_comfort_youtuber/",
 "https://www.reddit.com/r/AskReddit/comments/1hpjwjf/left_handed_people_what_in_the_world_just_doesnt/",
  "https://www.reddit.com/r/AskReddit/comments/1i0g0yz/pew_research_nearly_half_us_adults_say_dating_has/", 
    "https://www.reddit.com/r/AskReddit/comments/1hrrufe/whats_one_historical_fact_that_they_wont_teach/",
    "https://www.reddit.com/r/AskReddit/comments/1hvqz4d/whats_a_life_hack_so_good_you_almost_dont_want_to/",
    "https://www.reddit.com/r/AskReddit/comments/1hum6a0/ex_prisoners_of_reddit_what_is_something_about/",
    "https://www.reddit.com/r/AskReddit/comments/1i807vk/whats_a_random_life_hack_you_swear_by_even_if_no/",
   "https://www.reddit.com/r/AskReddit/comments/1hsrq4x/when_did_you_realize_youre_a_horrible_person/" 
    
]

asset_dir = "/mnt/n/data/reddit-yt-tts/queued"
completed_dir="/mnt/n/data/reddit-yt-tts/completed"


# In[5]:


submission_titles = [x.split("/")[-1] for x in glob.glob(f"{asset_dir}/*")]


# In[3]:



# In[6]:


completed_metadata_files = [x.split("/")[-1] for x in glob.glob(f"{asset_dir}/../assets/*metadata*/*")]


# In[7]:


skipped_threads = ["bernie_sanders_says_us_should_confiscate_100_of"]


# In[8]:


for submission_title in submission_titles:
    
    #clean submission_title
    submission_title = submission_title.replace("â\x80\x99","'")
    print(f"Processing {submission_title} ...")
    post_url = list(filter(lambda x: submission_title in x,post_urls))[0] #index errors due to missing post_url 
    
    #skip completed files
    completed_metadata_files = [x.split("/")[-1].replace(".json","") for x in glob.glob(f"{asset_dir}/../assets/*metadata*/*")]
    if submission_title in completed_metadata_files:
        continue
        
    if submission_title in skipped_threads:
        continue
    thread_ids = [x.split("/")[-1] for x in glob.glob(f"{asset_dir}/{submission_title}/*")]

    def build_thread_vid(thread_id,bg_img="assets/Qv79akqGQt0.png"):
        images = glob.glob(f"{asset_dir}/{submission_title}/{thread_id}/*.png")
        images = sorted(images)
        audio_file_dir = f"{completed_dir}/{submission_title}/{thread_id}"
        # https://zulko.github.io/moviepy/examples/examples.html
        clips = []

        audio_clips = []
        total_duration = 0
        transition_duration = 0

        for i,image in enumerate(images):

            image_file = image.split("/")[-1]
            audio_file = image_file.replace(".png",".wav")
            try:
                audio_clip = mpy.AudioFileClip(f"{audio_file_dir}/{audio_file}").set_start(total_duration)
            except OSError as e:
                print(f"audio file {audio_file_dir}/{audio_file} not found")
                continue
            audio_clips.append(audio_clip)
            audio_duration = audio_clip.duration

            clip = mpy.ImageClip(image).set_duration(audio_duration).set_start(total_duration).set_position(("center", "center"))

            bg_clip = mpy.ImageClip(bg_img).set_duration(audio_duration).set_start(total_duration)
            if clip.size[0] > bg_clip.size[0] * .9:
                clip = clip.resize((bg_clip.size[0]*.9,clip.size[1]*.9))

            if clip.size[1] > bg_clip.size[1] * .9:
                clip = clip.resize((clip.size[0]*.9,bg_clip.size[1]*.9))
            #clip = clip.set_start(5) # TODO: change duration based on audio
            #clip = clip.resize((1080,1920)) #TODO add tts audio
            clips.append(bg_clip)
            clips.append(clip)

            total_duration += audio_duration + transition_duration

        if len(clips) > 0 :
            final_audio_clip = mpy.CompositeAudioClip(audio_clips)
            final_clip = mpy.CompositeVideoClip(clips)
            final_clip.audio = final_audio_clip
            final_clip.write_videofile(f"{asset_dir}/{submission_title}/{thread_id}/thread_clip.mp4",fps=30,logger=None)


    #import moviepy.video.fx as vfx
    def build_intro_vid(bg_img):
        #intro
        intro_duration = 4   
        bg_clip = mpy.ImageClip(bg_img).set_duration(intro_duration)
        bg_clip = mpy.vfx.fadein(bg_clip,intro_duration-1,[0,0,0])
        bg_clip.write_videofile(f"{asset_dir}/{submission_title}/intro.mp4",fps=30,logger=None)

    def build_outro_vid(bg_img):
        #outro
        outro_duration = 4   
        bg_clip = mpy.ImageClip(bg_img).set_duration(outro_duration)
        bg_clip = mpy.vfx.fadeout(bg_clip,outro_duration-1,[0,0,0])
        bg_clip.write_videofile(f"{asset_dir}/{submission_title}/outro.mp4",fps=30,logger=None)

    thread_ids = sorted(list(set(thread_ids)))

    f"{asset_dir}/{submission_title}/{thread_ids[0]}"

    bg_img="assets/bg_images/Qv79akqGQt0.png"

    # save threads to clips
    # vote_div.text

    thread_ids = sorted(list(set(thread_ids)))
    for thread in tqdm(thread_ids):
        print(thread)
        build_thread_vid(thread,bg_img)


    build_intro_vid(bg_img)
    build_outro_vid(bg_img)

    # save c to 

    videos = glob.glob(f"{asset_dir}/{submission_title}/*/*.mp4")
    videos = sorted(list(set(videos)))

    #prepend intro
    intro = glob.glob(f"{asset_dir}/{submission_title}/intro.mp4")
    videos = intro + videos

    #append outro
    outro = glob.glob(f"{asset_dir}/{submission_title}/outro.mp4")
    videos = videos + outro



    #videos

    all_video_clips = [mpy.VideoFileClip(video) for video in videos]

    videos_w_sound_corrected = []
    for clip in all_video_clips:
        if clip.audio:
            clip.audio = clip.audio.fx(mpy.afx.audio_normalize)
        videos_w_sound_corrected.append(clip)


    transition = mpy.VideoFileClip("assets/tvstatic-180-720.mp4")

    transition2 = mpy.VideoFileClip("assets/free-vhs-overlay-1080-720.mp4")

    transition2.audio = transition.audio
    transition2 = transition2.set_duration(.7)
    #transition = transition.resize(1080,720) resizing crashes kernal reesizing 
    #using ffmpeg on cli instead:  https://trac.ffmpeg.org/wiki/Scaling

    #split bg music into parts: https://unix.stackexchange.com/questions/280767/how-do-i-split-an-audio-file-into-multiple
    #ffmpeg -i somefile.mp3 -f segment -segment_time 3 -c copy out%03d.mp3


    final_video = mpy.concatenate_videoclips(videos_w_sound_corrected,method="compose",transition=transition2)

    def get_audio(v_duration):
        import random
        get_audio = True
        sel_file_clips = []
        sel_file_names = []
        while get_audio:

            audio_files = glob.glob("assets/bg_music/*.opus")

            sel_file = random.choice(audio_files)

            sel_file_clips.append(mpy.AudioFileClip(sel_file))
            sel_file_names.append(sel_file)

            all_duration = [ x.duration for x in sel_file_clips]

            total_duration = 0

            for duration in all_duration:
                total_duration += duration

            if total_duration >= v_duration:
                get_audio = False

        out_audio = mpy.concatenate_audioclips(sel_file_clips).set_duration(v_duration)
        return out_audio, sel_file_names 


    #import moviepy.video.fx as vfx
    #import moviepy.audio.fx as afx
    bg_music,bg_music_files = get_audio(final_video.duration)

    bg_music = bg_music.fx( mpy.afx.audio_normalize).fx( mpy.afx.volumex, 0.3).fx( mpy.afx.audio_fadein, 1.0)
    tts_audio = final_video.audio

    bgm_w_tts_audio = mpy.CompositeAudioClip([bg_music,tts_audio])

    final_video.audio = bgm_w_tts_audio

    final_video.write_videofile(f"assets/final_videos/{submission_title}.mp4",fps=30,logger=None)

    # Save video credits
    def check_wildcard(file,credit):
        prefix = credit["filename"].split(".*")[0]
        return prefix in file

    def get_audio_credits(audio_files):
        import yaml
        with open("assets/bg_music/asset_credits.yml","r") as f:
            credits = yaml.load(f,yaml.FullLoader)
        active_credits = []
        for credit in credits["assets"]:
            for file in audio_files:
                if ".*" in credit["filename"] and check_wildcard(file,credit):
                    active_credits.append(credit)
                elif credit["filename"] == file:
                    active_credits.append(credit)
        return active_credits

    credits = get_audio_credits(bg_music_files)
    all_credit_texts = []
    for credit in credits:
        credit_text = f"""
        - {credit['artist']} - {credit['title']} 
        - Provided by Lofi Girl
        - Watch: {credit['link']}
        - Listen: {credit['spotify']}
        """
        all_credit_texts.append(credit_text)
    audio_credits = "--------\n".join(set(all_credit_texts))
    #print(audio_credits)

    with open(f"assets/final_video_credits/{submission_title}.txt","w") as f:
        f.write(audio_credits)

    glob.glob(f"{asset_dir}/../archive_text/{submission_title}/*")

    from make_thumbnail import make_thumbnail_img
    prompt_file = glob.glob(f"{asset_dir}/../archive_text/{submission_title}/0001*/0000.txt")[0]

    with open(prompt_file,"r",encoding="latin1") as f:
        prompt_text = f.read()
    #prompt_text = "What did the pandemic ruin more than we realise?"    
    make_thumbnail_img(prompt_text,f"assets/final_video_thumbnails/{submission_title}.png",font="assets/fonts/limejuice/Limejuice.ttf")

    prompt_text

    import json

    metadata = dict(prompt_text=prompt_text
                    ,submission_title=submission_title
                    ,video_file=f"assets/final_videos/{submission_title}.mp4"
                    ,credits_file=f"assets/final_video_credits/{submission_title}.txt"
                   ,thumbnail=f"assets/final_video_thumbnails/{submission_title}.png"
                   ,post_url=post_url)

    with open(f"assets/final_video_metadata/{submission_title}.json","w") as f:
        json.dump(metadata,f)


    print(f"done!...info:\n prompt_text: {prompt_text}\n submission_title: {submission_title} \n video_file: assets/final_videos/{submission_title}.mp4 \n credits_file: assets/final_video_credits/{submission_title}.txt\n thumbnail: assets/final_video_thumbnails/{submission_title}.png \n post: {post_url}")


# In[20]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




