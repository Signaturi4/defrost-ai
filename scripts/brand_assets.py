"""Brand assets from docs/brand/source-artwork.jpg: cut the tile out of its dark backdrop, rebuild a full-bleed square
app icon (HIG app-icons: square, unmasked layers; the system applies the corner mask), then derive the masked logo,
favicons, the apple-touch icon and the GitHub social preview (1280x640, BRAND.md colors, Geist type).

    uv run --no-project --with pillow --with numpy python scripts/brand_assets.py --fonts <dir with geist-sans/ geist-mono/>

Geist is OFL; get it with `npm pack geist` (package/dist/fonts)."""
import argparse
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument("--fonts", required=True); args=ap.parse_args()
OUT=str(Path(__file__).resolve().parents[1] / "docs/brand") + "/"
F=args.fonts.rstrip("/") + "/"
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
SRC=OUT+'source-artwork.jpg'
src=np.asarray(Image.open(SRC).convert('RGB')).astype(float)
x0,y0,x1,y1=147,146,877,877; R=178
H,W=src.shape[:2]; yy,xx=np.mgrid[0:H,0:W]
def rr(inset):
    m=Image.new('L',(W,H),0); ImageDraw.Draw(m).rounded_rectangle([x0+inset,y0+inset,x1-inset,y1-inset],radius=max(R-inset,1),fill=255); return np.asarray(m)>0
r,g,b=src[...,0],src[...,1],src[...,2]
gl=np.asarray(Image.fromarray(((r>160)*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(41)))>0
bgpix=rr(14)&(b>200)&(b-r>90)&~gl
U=(xx-(x0+x1)/2)/((x1-x0)/2); V=(yy-(y0+y1)/2)/((y1-y0)/2)
def basis(U,V):
    D=np.sqrt(U**2+(V+1.0)**2)
    return np.stack([np.ones_like(U),U,V,U*U,U*V,V*V,D,D*D],-1)
A=basis(U,V); coefs=[np.linalg.lstsq(A[bgpix],src[...,c][bgpix],rcond=None)[0] for c in range(3)]
def bg(N, scale):
    v,u=np.mgrid[0:N,0:N]; u=np.clip((u/(N-1)*2-1)/scale,-1,1); v=np.clip((v/(N-1)*2-1)/scale,-1,1)
    B=basis(u,v); return np.stack([B@c for c in coefs],-1)
fit=np.stack([A@c for c in coefs],-1)
m=Image.fromarray((rr(18)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)); a=(np.asarray(m)/255.)[...,None]
tile=(src*a+fit*(1-a))[y0:y1,x0:x1]
N=1024; S=0.84
n=int(N*S)
content=np.asarray(Image.fromarray(np.clip(tile,0,255).astype(np.uint8)).resize((n,n),Image.LANCZOS)).astype(float)
canvas=bg(N,S); off=(N-n)//2
fm=Image.new('L',(n,n),0); ImageDraw.Draw(fm).rounded_rectangle([40,40,n-40,n-40],radius=150,fill=255)
fm=(np.asarray(fm.filter(ImageFilter.GaussianBlur(18)))/255.)[...,None]
reg=canvas[off:off+n,off:off+n]; canvas[off:off+n,off:off+n]=content*fm+reg*(1-fm)
icon=Image.fromarray(np.clip(canvas,0,255).astype(np.uint8)); 
def squircle(N,ss=4,k=4.6):
    M=N*ss; v,u=np.mgrid[0:M,0:M]; u=np.abs((u+.5)/M*2-1); v=np.abs((v+.5)/M*2-1)
    return Image.fromarray(((u**k+v**k)<=1).astype(np.uint8)*255).resize((N,N),Image.LANCZOS)
logo=icon.convert('RGBA'); logo.putalpha(squircle(1024)); 

icon.save(OUT+'app-icon-1024.png', optimize=True)                        # square, unmasked (Icon Composer / stores)
icon.resize((180,180),Image.LANCZOS).save(OUT+'apple-touch-icon.png', optimize=True)  # iOS masks it itself
for s in (512,256): logo.resize((s,s),Image.LANCZOS).save(OUT+f'logo-{s}.png', optimize=True)
logo.resize((32,32),Image.LANCZOS).save(OUT+'favicon-32.png', optimize=True)
logo.save(OUT+'favicon.ico', sizes=[(16,16),(32,32),(48,48),(64,64)])
INK=(11,15,20); FROST=(244,246,248); MUTED_D=(154,164,178); AMBER=(245,165,36)
def social(path):
    W,H=1280,640; im=Image.new('RGB',(W,H),INK); d=ImageDraw.Draw(im)
    s=300; x,y=120,(H-s)//2
    glow=Image.new('RGBA',(W,H),(0,0,0,0)); gd=ImageDraw.Draw(glow)
    gd.rounded_rectangle([x+20,y+30,x+s-20,y+s+10],radius=70,fill=(47,111,235,120))
    im.paste(glow.filter(ImageFilter.GaussianBlur(40)),(0,0),glow.filter(ImageFilter.GaussianBlur(40)))
    l=logo.resize((s,s),Image.LANCZOS); im.paste(l,(x,y),l)
    tx=x+s+80
    wm=ImageFont.truetype(F+'geist-mono/GeistMono-Medium.ttf',88)
    tg=ImageFont.truetype(F+'geist-sans/Geist-SemiBold.ttf',46)
    st=ImageFont.truetype(F+'geist-sans/Geist-Regular.ttf',28)
    stb=ImageFont.truetype(F+'geist-mono/GeistMono-SemiBold.ttf',28)
    d.text((tx,198),'defrost-ai',font=wm,fill=FROST)
    d.text((tx,318),'Docs your AI actually reads.',font=tg,fill=FROST)
    d.text((tx,402),'94%',font=stb,fill=AMBER)
    w=d.textlength('94% ',font=stb)
    d.text((tx+w,402),'vs 6%: answering doc section in context',font=st,fill=MUTED_D)
    d.text((tx,446),'Local · free · open source · MCP',font=st,fill=MUTED_D)
    im.save(path, optimize=True)
social(OUT+'social-preview.png')
