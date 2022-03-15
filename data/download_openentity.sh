wget -O data.tar.gz http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz
tar zxvf data.tar.gz
mkdir openentity
cp release/crowd/dev.json openentity/dev.json
cp release/crowd/train.json openentity/train.json
cp release/crowd/test.json openentity/test.json
# cp release/distant_supervision/el_dev.json openentity/el_dev.json
# cp release/distant_supervision/el_train.json openentity/el_train.json
# cp release/distant_supervision/headword_dev.json openentity/headword_dev.json
# cp release/distant_supervision/headword_train.json openentity/headword_train.json
cp release/ontology/types.txt openentity/types.txt

