import { Component, ElementRef, NgZone, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { environment } from '../../environments/environment';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-dashboard',
  imports: [FormsModule, ReactiveFormsModule, CommonModule, MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    ReactiveFormsModule,],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {

ngOnInit(): void {
  //Called after the constructor, initializing input properties, and the first call to ngOnChanges.
  //Add 'implements OnInit' to the class.
  
}
  // Inject the UserDataService to get and update user data
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;

  @ViewChild('emotionDiv') emotionDiv!: ElementRef<HTMLDivElement>;


  detectedEmotion = '';

  errorMessage = '';

  showPermissionWarning = false;

  showPlaylist = false;

  songs: any[] = [];

  private stream: MediaStream | null = null;


  constructor(private http: HttpClient, private ngZone: NgZone) { }


  async startDetection() {

    try {

      this.stream = await navigator.mediaDevices.getUserMedia({ video: true });

      this.videoElement.nativeElement.srcObject = this.stream;

      this.errorMessage = '';

      this.detectEmotion();

    } catch (error) {

      console.error('Error accessing webcam: ', error);

      this.ngZone.run(() => {

        this.detectedEmotion = 'Error accessing webcam. Please check permissions.';

        this.errorMessage = 'Error accessing webcam. Please allow camera access.';

        this.showPermissionWarning = true;

      });

    }

  }


  private async detectEmotion() {

    const canvas = document.createElement('canvas');

    const context = canvas.getContext('2d');

    const video = this.videoElement.nativeElement;


    canvas.width = video.videoWidth;

    canvas.height = video.videoHeight;


    context?.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/png');


    try {

      const response = await this.http.post<any>('http://localhost:5000/api/detect_emotion', { image: imageData }).toPromise();



      this.ngZone.run(() => {

        if (this.stream) {

          this.stream.getTracks().forEach(track => track.stop());

          this.videoElement.nativeElement.srcObject = null;

        }


        this.detectedEmotion = `Detected Emotion: ${response.emotion}`;



        if (response.songs?.length) {

          this.songs = response.songs;

          this.showPlaylist = true;

        } else {

          this.showPlaylist = false;

          this.detectedEmotion += ' (No songs found for this emotion)';

        }

      });

    } catch (error) {

      console.error(error);

      this.ngZone.run(() => {

        this.detectedEmotion = 'Error in emotion detection. Try again.';

      });

      this.detectEmotion();

    }

  }


  getSpotifyEmbedUrl(url: string): string {

    const trackId = url.split('/').pop();

    return `https://open.spotify.com/embed/track/${trackId}`;

  }
}
