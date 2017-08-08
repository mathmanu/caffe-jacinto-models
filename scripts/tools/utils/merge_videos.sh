ffmpeg \
-start_number 0 -i /user/a0393608/files/datahdd/datasets/object-detect/ti/road-drive/ti-virb/V008_2015jul_VIRB0008_7m_6000fr/%06d.png \
-start_number 0 -i "/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/vision-dl-src/apps/segmentation/output/ti-virb5/%06d.png" \
-filter_complex "[0:v]crop=1280:640:320:220[left]; [1:v]crop=1280:640:320:220[right]; [left][right]hstack" -crf 16 output.mp4
