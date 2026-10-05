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
                # Thin the recorded joint dots into smooth centerlines before thickening.
                linked=cv2.morphologyEx(skel,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
                thin=cv2.ximgproc.thinning(linked)
                binary=(thin>0).astype(np.uint8)
                degree=cv2.filter2D(binary,-1,np.ones((3,3),np.uint8))-binary
                ey,ex=np.where((binary==1)&(degree==1))
                paths=[]
                for sx,sy in zip(ex,ey):
                    path=[(int(sx),int(sy))]; seen=set(path)
                    while len(path)<70:
                        cx,cy=path[-1]
                        options=[(nx,ny) for ny in range(max(0,cy-1),min(binary.shape[0],cy+2))
                                 for nx in range(max(0,cx-1),min(binary.shape[1],cx+2))
                                 if binary[ny,nx] and (nx,ny) not in seen]
                        if len(options)!=1: break
                        path.append(options[0]); seen.add(options[0])
                    if len(path)<5:
                        for px,py in path[:-1]: thin[py,px]=0
                    elif len(path)>12: paths.append(np.array(path,np.float32))
                scale=3
                size=(thin.shape[1]*scale,thin.shape[0]*scale)
                line_hi=cv2.resize(thin,size,interpolation=cv2.INTER_NEAREST)
                distance=cv2.distanceTransform(255-line_hi,cv2.DIST_L2,5)/scale
                distance=cv2.GaussianBlur(distance,(0,0),scale*.7)
                radius=float(np.clip(max(bw,bh)*.027,2.5,4.7))
                radii=np.full(distance.shape,radius,np.float32)
                Y,X=np.mgrid[:size[1],:size[0]].astype(np.float32)/scale
                for path in paths:
                    px,py=path[0]
                    d=np.sqrt((X-px)**2+(Y-py)**2)
                    radii=np.minimum(radii,radius*(.72+.28*np.minimum(d/(radius*3),1)))
                solid=np.uint8(distance<radii)*255
                # A proximal palm patch, blended smoothly into the digit surfaces.
                contours,_=cv2.findContours(linked,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                palm=np.zeros_like(linked)
                cv2.drawContours(palm,contours,-1,255,cv2.FILLED)
                yy,xx=np.where(linked>0)
                proximal=yy > (yy.min()+.46*(yy.max()-yy.min()))
                px,py=xx[proximal],yy[proximal]
                if len(px)>10:
                    keep=(px>=np.percentile(px,12)) & (px<=np.percentile(px,82))
                    points=np.column_stack([px[keep],py[keep]]).astype(np.int32)
                    if len(points)>3: cv2.fillConvexPoly(palm,cv2.convexHull(points),255)
                # Remove the original joint dots from the palm mask before merging.
                palm=cv2.morphologyEx(palm,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
                palm=cv2.resize(palm,size,interpolation=cv2.INTER_LINEAR)
                palm=cv2.GaussianBlur(palm,(0,0),scale*1.7)
                solid=np.maximum(solid,palm)
                smooth=cv2.GaussianBlur(solid.astype(np.float32)/255,(0,0),scale*.85)
                inside=np.uint8(smooth>.46)*255
                dist=cv2.distanceTransform(inside,cv2.DIST_L2,5)/scale
                # Rounded cross-section with a broad palm and smoother lighting normals.
                z=np.sqrt(np.maximum(0,radius**2-np.maximum(radius-dist,0)**2)) + .22*cv2.GaussianBlur(dist,(0,0),scale*3)
                z=cv2.GaussianBlur(z,(0,0),scale*1.05)
                gx=cv2.Sobel(z,cv2.CV_32F,1,0,ksize=3)*scale/8
                gy=cv2.Sobel(z,cv2.CV_32F,0,1,ksize=3)*scale/8
                norm=np.sqrt(gx*gx+gy*gy+1)
                nx,ny,nz=-gx/norm,-gy/norm,1/norm
                diffuse=np.clip(nx*-.38+ny*-.48+nz*.79,0,1)
                spec=np.maximum(0,nx*-.21+ny*-.26+nz*.942)**30
                rim=(1-nz)**2
                base=np.array(color,dtype=np.float32)*.72+np.array([190,200,205])*.28
                surface=base[None,None,:]*(.38+.62*diffuse[:,:,None])+45*spec[:,:,None]+9*rim[:,:,None]
                # Subtle nail caps and joint creases on unambiguous distal branches.
                details=np.zeros_like(inside)
                creases=np.zeros_like(inside)
                for path in paths:
                    if path[0,1]>yy.min()+.66*(yy.max()-yy.min()): continue
                    tip=path[0]; direction=path[min(9,len(path)-1)]-tip
                    length=np.linalg.norm(direction)
                    if length<3: continue
                    direction/=length
                    center=(tip+direction*radius*1.5)*scale
                    angle=float(np.degrees(np.arctan2(direction[1],direction[0])))
                    cv2.ellipse(details,tuple(np.int32(center)),(max(2,int(radius*1.1*scale)),max(2,int(radius*.57*scale))),angle,0,360,255,-1,cv2.LINE_AA)
                    for fraction in [.48,.78]:
                        k=int((len(path)-1)*fraction)
                        tangent=path[min(k+2,len(path)-1)]-path[max(0,k-2)]
                        tangent/=max(np.linalg.norm(tangent),1)
                        perp=np.array([-tangent[1],tangent[0]])
                        p1=(path[k]-perp*radius*.58)*scale; p2=(path[k]+perp*radius*.58)*scale
                        cv2.line(creases,tuple(np.int32(p1)),tuple(np.int32(p2)),255,scale,cv2.LINE_AA)
                nail=cv2.GaussianBlur(details.astype(np.float32)/255,(0,0),scale*.45)[:,:,None]
                crease=cv2.GaussianBlur(creases.astype(np.float32)/255,(0,0),scale*.36)[:,:,None]
                surface=surface*(1-.10*crease)
                surface=surface*(1-.20*nail)+np.minimum(base+48,255)[None,None,:]*.20*nail
                coverage=np.clip((smooth-.36)/.22,0,1)
                surface=cv2.resize(np.clip(surface,0,255),(skel.shape[1],skel.shape[0]),interpolation=cv2.INTER_AREA)
                alpha=cv2.resize(coverage,(skel.shape[1],skel.shape[0]),interpolation=cv2.INTER_AREA)[:,:,None]*blend
                # Remove baked strokes where the slimmer silhouette exposes them.
                repair=cv2.dilate(skel,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
                clean=cv2.inpaint(out[y0:y1,x0:x1],repair,3,cv2.INPAINT_TELEA)
                out[y0:y1,x0:x1]=np.uint8(out[y0:y1,x0:x1]*(1-blend)+clean*blend)
                roi=out[y0:y1,x0:x1].astype(np.float32)
                out[y0:y1,x0:x1]=np.uint8(roi*(1-alpha)+surface*alpha)
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='demo.mov'); p.add_argument('--output',default='outputs/tracking-surface-refined.mp4'); p.add_argument('--still',type=float); a=p.parse_args()
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
