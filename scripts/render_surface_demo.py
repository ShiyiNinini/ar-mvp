"""Image-space hand surface experiment from baked colored skeletons, not 3D pose recovery."""
import argparse, json, subprocess
from pathlib import Path
import cv2
import numpy as np


def render(frame, blend):
    h,w=frame.shape[:2]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    masks=[cv2.inRange(hsv,(35,100,90),(90,255,255)),cv2.inRange(hsv,(145,100,75),(179,255,255))]
    out=frame.copy()
    for mask,color in zip(masks,[(110,238,100),(183,98,240)]):
        mask[:60]=0
        mask[h-35:]=0
        # Process each camera independently, keeping the recorded stereo geometry.
        for lo,hi in [(0,w//2),(w//2,w)]:
            part=mask[:,lo:hi]
            count,labels,stats,_=cv2.connectedComponentsWithStats(part)
            for idx in range(1,count):
                x,y,bw,bh,area=stats[idx]
                if area<75 or bh<16: continue
                pad=20
                x0=max(lo+x-pad,lo); x1=min(lo+x+bw+pad,hi)
                y0=max(y-pad,60); y1=min(y+bh+pad,h-35)
                skel=np.uint8(labels[y0:y1,x0-lo:x1-lo]==idx)*255
                skel=cv2.morphologyEx(skel,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
                # Fill enclosed palm polygons while retaining open spaces between fingers.
                contours,_=cv2.findContours(skel,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                solid=np.zeros_like(skel)
                cv2.drawContours(solid,contours,-1,255,cv2.FILLED)
                # Approximate palm from proximal image-space strokes; no depth is inferred.
                yy,xx=np.where(skel>0)
                proximal=yy > (yy.min()+0.46*(yy.max()-yy.min()))
                px,py=xx[proximal],yy[proximal]
                if len(px)>10:
                    keep=(px>=np.percentile(px,12)) & (px<=np.percentile(px,82))
                    points=np.column_stack([px[keep],py[keep]]).astype(np.int32)
                    if len(points)>3: cv2.fillConvexPoly(solid,cv2.convexHull(points),255)
                radius=max(3,min(7,round(max(bw,bh)*0.035)))
                solid=cv2.dilate(solid,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2))
                dist=cv2.distanceTransform(solid,cv2.DIST_L2,5)
                height=np.sqrt(np.maximum(dist,0))
                height=cv2.GaussianBlur(height,(5,5),1)
                gx=cv2.Sobel(height,cv2.CV_32F,1,0,ksize=3)*0.3
                gy=cv2.Sobel(height,cv2.CV_32F,0,1,ksize=3)*0.3
                denom=np.sqrt(gx*gx+gy*gy+1)
                light=np.clip((-gx*-.45-gy*-.6+.66)/denom,0,1)
                shade=.36+.64*light
                surface=np.clip(np.array(color)[None,None,:]*shade[:,:,None]+24*light[:,:,None]**12,0,255)
                alpha=cv2.GaussianBlur(solid.astype(np.float32)/255,(3,3),.65)[:,:,None]*blend
                roi=out[y0:y1,x0:x1].astype(np.float32)
                out[y0:y1,x0:x1]=np.uint8(roi*(1-alpha)+surface*alpha)
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='demo.mov'); p.add_argument('--output',default='outputs/tracking-surface-demo.mp4'); p.add_argument('--still',type=float); a=p.parse_args()
    cap=cv2.VideoCapture(a.input)
    fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(3)); h=int(cap.get(4))
    if a.still is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC,a.still*1000); ok,frame=cap.read(); assert ok
        result=render(frame,1)
        cv2.imwrite(a.output,result); return
    enc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',a.output],stdin=subprocess.PIPE)
    i=0; written=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        t=i/fps; i+=1
        if int((i-1)*30/fps)<written: continue
        blend=float(np.clip((t-5)/1.2,0,1)); blend=blend*blend*(3-2*blend)
        result=render(frame,blend) if blend else frame
        label='SKELETON' if blend==0 else ('SURFACE TRANSITION' if blend<1 else 'SURFACE - IMAGE-BASED APPROXIMATION')
        cv2.rectangle(result,(w//2-225,12),(w//2+225,43),(22,25,28),-1)
        cv2.putText(result,label,(w//2-212,33),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1,cv2.LINE_AA)
        enc.stdin.write(result.tobytes()); written+=1
        if written%150==0: print(f'{written/30:.0f}s rendered',flush=True)
    cap.release(); enc.stdin.close(); assert enc.wait()==0
    print(json.dumps({'frames':written,'fps':30,'width':w,'height':h,'output':a.output}),flush=True)

if __name__=='__main__': main()
