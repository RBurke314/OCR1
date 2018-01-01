
"""jpgtxt = open(r'C:\Users\Rob\dev\VisionSystems\OCR\Img_Cap.jpg','rb').read().encode('base64').replace('\n','')

f = open("jpg1_b64.txt", "w")
f.write(jpgtxt)
f.close()"""

# ----
"""newjpgtxt = open("jpg1_b64.txt","rb").read()

g = open("Img_Cap_blank3.jpg", "w")
g.write(newjpgtxt.decode('base64'))
g.close()"""
newjpgtxt = open("jpg1_b64.txt","rb").read()
fh = open("Img_Cap_blank4.jpg", "wb")
fh.write(newjpgtxt.decode('base64jpg'))
fh.close()


