img0=readfits('m31bK.fits')
img1=readfits('m31bK30_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bK30_scl.fits',img1*scl

img0=readfits('m31bH.fits')
img1=readfits('m31bH30_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bH30_scl.fits',img1*scl

img0=readfits('m31bJ.fits')
img1=readfits('m31bJ30_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bJ30_scl.fits',img1*scl
 
img0=readfits('m31bK20_2.fits')
img1=readfits('m31bK20_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bK20_scl.fits',img1*scl

img0=readfits('m31bH20_2.fits')
img1=readfits('m31bH20_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bH20_scl.fits',img1*scl

img0=readfits('m31bJ20_2.fits')
img1=readfits('m31bJ20_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bJ20_scl.fits',img1*scl

img0=readfits('m31bK50.fits')
img1=readfits('m31bK50_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bK50_scl.fits',img1*scl

img0=readfits('m31bH50.fits')
img1=readfits('m31bH50_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bH50_scl.fits',img1*scl

img0=readfits('m31bJ50.fits')
img1=readfits('m31bJ50_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bJ50_scl.fits',img1*scl

img0=readfits('m31bK100.fits')
img1=readfits('m31bK100_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bK100_scl.fits',img1*scl

img0=readfits('m31bH100.fits')
img1=readfits('m31bH100_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bH100_scl.fits',img1*scl

img0=readfits('m31bJ100.fits')
img1=readfits('m31bJ100_avg.fits')
scl=total(img0)/total(img1)
writefits,'m31bJ100_scl.fits',img1*scl



end
