$(document).ready (function () {
        
    // DOM elements
    var body = $('html')
    var html = $('html')
    var spinner = $('.loader')
    var container = $('.main')
    var wrapper = $('#wrapper')
        
    var setLoading = function () {
        spinner.removeAttr ('hidden');
        container.attr ('hidden', true);
        isLoading = true;
    }
        
    var removeLoading = function () {
        spinner.attr ('hidden', true);
        container.removeAttr ('hidden');
        isLoading = false;
    }
    
    // Remove loading
    removeLoading ()
    
    
    
    // Select2
    wrapper.find ('select').select2 ();


    /** @var audio DOM */
    var audio = document.querySelector ("#audio");


    /** @var play DOM */
    var play  = document.querySelector ("#play");
    play.recording = false;
    play.count = 0;
    play.onclick = () => {
      
        // Empty results
        wrapper.find ('.text').find ('.ph').html ('')
        wrapper.find ('.multimodal').find ('.ph').html ('')
        wrapper.find ('textarea').val ('')
        
        
        // Add recording
        play.classList.add ("recording-state");
        
        
        
        // First time button is pressed
        if (play.count === 0) {
            
            // Start recording
            navigator.mediaDevices.getUserMedia ({ audio: true, video: false }).then ((stream) => {
                let recordedChunks = [];
                const mediaRecorder  = new MediaRecorder(stream, {mimeType: "audio/webm"});
                play.mediaRecorder  = mediaRecorder;
                play.recordedChunks = recordedChunks;

                mediaRecorder.addEventListener('dataavailable', function(e) {
                    if (e.data.size > 0) {
                        recordedChunks.push(e.data);
                    }
                });
                
                
                // Audio format WebM
                mediaRecorder.addEventListener ('stop', function () {
                    
                    play.classList.remove ("recording-state");
                    audio.src = URL.createObjectURL (new Blob(recordedChunks));
                    let audio_field = wrapper.find('[name="audio"]');
                    var reader = new window.FileReader ();
                    reader.readAsDataURL (new Blob(recordedChunks));
                    reader.onloadend = function() {
                        base64 = reader.result;
                        audio_field.val(base64);
                    }
                    recordedChunks = [];
                });

                mediaRecorder.start();
            });
            play.recording = true;
            play.count = 1;
            return ;
        }

        if (play.recording) {
            play.mediaRecorder.stop ();
            play.recording = false;
        } else {
            play.mediaRecorder.start ();
            play.recording = true;
        }
        play.count++;
    }


    // Files
    var dropZone = wrapper.find ("#drop-zone");
    var fileInput = wrapper.find ("#audio-input");
    var fileName = wrapper.find ("#file-name");


    /** @var allowedTypes Array */
    const allowedTypes = [
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
    ];


    /** @var allowedTypes Array */
    const allowedExtensions = [".wav", ".mp3"];

    /**
     * handleFile
     */

    function handleFile (file) {
        
        // No file
        if ( ! file) {
            return;
        }
        
        
        /** @var isValidType */
        const isValidType = allowedTypes.includes (file.type);
        const isValidExtension = allowedExtensions.some (ext => name.endsWith(ext));

        if ( ! isValidType &&  ! isValidExtension) {
            vex.dialog.alert ("Please, audio file must be WAV or MP3");
            fileInput.val ("");
            return;
        }


        fileName.textContent = `File: ${file.name}`;
        
    }


    // Events
    dropZone.on ("click", function (e) {
        e.preventDefault ();
        fileInput.trigger ("click");
    });

    fileInput.on ("change", function () {
        handleFile (this.files[0]);
    });


    // Drag / drop
    dropZone.on ("dragover", function (e) {
        e.preventDefault ();
        e.stopPropagation ();
        dropZone.addClass ("dragover");
    });

    dropZone.on ("dragleave", function (e) {
        e.preventDefault ();
        e.stopPropagation ();
        dropZone.removeClass ("dragover");
    });

    dropZone.on ("drop", function (e) {
        e.preventDefault ();
        e.stopPropagation ();

        dropZone.removeClass ("dragover");

        const file = e.originalEvent.dataTransfer.files[0];
        handleFile (file);
    });


    /** @var form Object */
    var form = wrapper.find ('form[name="emotion"]');


    // Form submission
    form.on ('submit', function (e) {
        
        // Prevent default
        e.preventDefault ();
        
        
        /** @var data Object */
        var data = {};
        $.each (form.serializeArray (), function () {
            data[this.name] = this.value;
        });
        
        
        // Check values
        var error = false;
        switch (data.mode) {
            case 'text':
                error = data.transcription ? false : true;
                break;
                
            case 'audio':
                error = data.audio ? false : true;
                break;
            
            // Multmiodal
            default:
                error = ! (data.transcription && data.audio);
                break;
        }
        
        
        // Display
        if (error) {
            vex.dialog.alert ("No audio nor text")
            return;
        }
        
        
        // Set loading
        setLoading ();
        
        
        // Get call
        $.ajax ({
            url: "/predict",
            method: "POST",
            data: JSON.stringify (data),
            processData: false,
            contentType: "application/json; charset=utf-8",
            success: function (response) {
                
                // Remove
                removeLoading ();
                
                
                /** @var emotions */
                var emotions = response.scores;

                
                // Sort
                emotions = Object.fromEntries (
                    Object.entries (emotions).sort (([,a],[,b]) => b - a)
                );
                

                // Build HTML
                html_text = '';
                $.each (emotions, function (emotion, val) {
                    html_text += '<div class="emotion-bar"><label for="mm-' + emotion + '">' + emotion + '</label><progress id="mm-' + emotion + '" title="' + emotion + '" data-name="' + emotion + '" class="' + emotion + '" max="1" value="' + val + '"></progress></div>'
                });
                
                
                wrapper.find ('.text').find ('.ph').html (html_text)
            },
            error: function (xhr) {
                console.error("Error:", xhr);
            }
        });
    });


    // Remove loading state
    body.removeClass ('loading-state');
            
});