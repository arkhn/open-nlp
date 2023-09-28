gcloud storage cp $1 gs://glossaries-translation-api/
filename=`basename $1`
curl -X POST \
    -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d '{
        "name":"projects/leafy-ember-294410/locations/us-central1/glossaries/'$2'",
        "languagePair": {
          "sourceLanguageCode": "en",
          "targetLanguageCode": "fr"
        },
      "inputConfig": {
        "gcsSource": {
          "inputUri": "gs://glossaries-translation-api/'$filename'"
        }
      }
    }' \
    "https://translation.googleapis.com/v3/projects/leafy-ember-294410/locations/us-central1/glossaries"
