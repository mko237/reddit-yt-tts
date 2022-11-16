# MAIN CELL (RUN THIS)
from PIL import Image, ImageDraw, ImageFont
from image_utils import ImageText

def make_thumbnail_img(reddit_txt,filename):
    # image import
    #image = Image.open('picture.jpg')
    #draw = ImageDraw.Draw(image) # creates a draw object 
    yt_thumbnail_res = (1280,720) #https://support.google.com/youtube/answer/72431?hl=en
    image = Image.new('RGB', yt_thumbnail_res)
    draw = ImageDraw.Draw(image) 

    # draw boarder
    draw.rectangle((200,400,700,700), fill = (255,0,0), outline = 'yellow', width = 5)

    #reddit_txt = "What is the stupidest thing a large amount of people believe in?"

    # draw text 
    color = (255, 255, 255)
    yt_thumbnail_res = (1280,720) #https://support.google.com/youtube/answer/72431?hl=en
    text = reddit_txt
    font = 'assets/Halogen.ttf'
    fnt_size = 80 # may need to update this depending on title length
    w_pad= 5
    img = ImageText(yt_thumbnail_res, background=(0, 0, 0, 255))
    img.draw.rectangle((0,0,1280,720),fill= (0,0,0), outline = 'white', width = 18)
    # img.write_text(50, 50, "(/r/AskReddit)", font_filename=font,
    #               font_size='fill', max_height=60, color=color)

    sub_text_size = img.write_text_box(0, 0,"(/r/AskReddit)", box_width=yt_thumbnail_res[0]-w_pad, font_filename=font,
                       font_size=int(fnt_size*1.1), color=color, place='center',justify_last_line=True)


    main_text_size = img.write_text_box(0, 182, text, box_width=yt_thumbnail_res[0]-w_pad, font_filename=font,
                       font_size=fnt_size, color=color, place='center',justify_last_line=True)
    main_text_y_pos = (yt_thumbnail_res[-1]/2) - main_text_size[-1]/2
    sub_text_y_pos = main_text_y_pos - (w_pad*30)

    #recreating image with text centered
    img = ImageText(yt_thumbnail_res, background=(0, 0, 0, 255))
    img.draw.rectangle((0,0,1280,720),fill= (0,0,0), outline = 'white', width = 18)
    # img.write_text(50, 50, "(/r/AskReddit)", font_filename=font,
    #               font_size='fill', max_height=60, color=color)


    img.write_text_box(0, sub_text_y_pos,"(/r/AskReddit)", box_width=yt_thumbnail_res[0]-w_pad, font_filename=font,
                       font_size=int(fnt_size*1.1), color=color, place='center',justify_last_line=True)

    img.write_text_box(0, main_text_y_pos, text, box_width=yt_thumbnail_res[0]-w_pad, font_filename=font,
                       font_size=fnt_size, color=color, place='center',justify_last_line=True)
    
    img.save(filename)