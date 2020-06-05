from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

##APIキーの情報
key = "公開鍵"
secret = "秘密鍵"
#連続で情報を送るとサーバに影響が出るため待ち時間を入れる。1秒おき
wait_time = 1

##保存フォルダの指定
#Pythonプログラムを呼び出す時の名前を使ってフォルダ名にする
#コマンドラインの入力の2番目の情報を代入。CurrentDirectoryの下にあるanimalnameの下に保存
animalname = sys.argv[1]
savedir = "./" + animalname

#検索キーワードはanimalname
#per_pageは何件取得するか。300ほどほしいので、多めに400とする。sortは関連順
flickr = FlickrAPI(key, secret, format="parsed-json")
result = flickr.photos.search(
    text = animalname,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence'
)

photos = result['photos']
#返り値を表示する
#pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    #重複していればスキップする、していなければダウンロードする
    if os.path.exists(filepath): continue
    urlretrieve(url_q,filepath)
    time.sleep(wait_time)
