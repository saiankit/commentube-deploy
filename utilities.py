import urllib
import urllib.request
import json
def generate_image_link(url):
    return "https://img.youtube.com/vi/" + generate_video_id(url) + "/maxresdefault.jpg"

def generate_video_id(url):
    parsed_url = urllib.parse.urlparse(url)
    video_id = parsed_url.query.split("=")[1]
    return video_id
   
def generate_text(url):
    #change to yours VideoID or change url inparams
    VideoID =  generate_video_id(url)
    
    params = {"format": "json", "url": "https://www.youtube.com/watch?v=%s" % VideoID}
    url = "https://www.youtube.com/oembed"
    query_string = urllib.parse.urlencode(params)
    url = url + "?" + query_string
    
    with urllib.request.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
        return data['title']
