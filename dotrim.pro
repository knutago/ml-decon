sz0=size(readfits('m31bJ.fits'))
img=readfits('m31bJ30_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bJ30_trim.fits',imgo

sz0=size(readfits('m31bH.fits'))
img=readfits('m31bH30_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bH30_trim.fits',imgo

sz0=size(readfits('m31bK.fits'))
img=readfits('m31bK30_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bK30_trim.fits',imgo


sz0=size(readfits('m31bJ20_2.fits'))
img=readfits('m31bJ20_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bJ20_trim.fits',imgo

sz0=size(readfits('m31bH20_2.fits'))
img=readfits('m31bH20_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bH20_trim.fits',imgo

sz0=size(readfits('m31bK20_2.fits'))
img=readfits('m31bK20_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bK20_trim.fits',imgo



sz0=size(readfits('m31bJ50.fits'))
img=readfits('m31bJ50_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bJ50_trim.fits',imgo

sz0=size(readfits('m31bH50.fits'))
img=readfits('m31bH50_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bH50_trim.fits',imgo

sz0=size(readfits('m31bK50.fits'))
img=readfits('m31bK50_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bK50_trim.fits',imgo


sz0=size(readfits('m31bJ100.fits'))
img=readfits('m31bJ100_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bJ100_trim.fits',imgo

sz0=size(readfits('m31bH100.fits'))
img=readfits('m31bH100_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bH100_trim.fits',imgo

sz0=size(readfits('m31bK100.fits'))
img=readfits('m31bK100_obs2.fits')
sz1=size(img)
dsz=sz1-sz0
imgo=img[dsz[1]/2:dsz[1]/2+sz0[1]-1,dsz[2]/2:dsz[2]/2+sz0[2]-1]
writefits,'m31bK100_trim.fits',imgo





end
