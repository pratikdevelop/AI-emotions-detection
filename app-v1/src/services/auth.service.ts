import { Injectable, signal, resource } from '@angular/core';
import { environment } from '../environments/environment';

// Define types
interface SignupRequest {
  email: string;
  password: string;
  name?: string;
}

interface SignupResponse {
  user: {
    id: string;
    email: string;
    name?: string;
  };
  token: string;
}

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  // Signal to hold the signup payload (updated by the component)
  private signupPayload = signal<SignupRequest>({
    email: '',
    password: '',
    name: '',
  });

  // Define the signup resource
  signupResource = resource<any,any>({
    request: () => this.signupPayload(), // Reactive to payload changes
    loader: async ({ request, abortSignal }) => {
      const response = await fetch(`${environment.api}/signup`, {
        method: 'POST',
        signal: abortSignal,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error('Unable to sign up!');
      }

      return await response.json();
    },
  });

  // Method to update the payload and trigger the resource
  signup(payload: any) {
    this.signupPayload.set({ email: payload.email, password: payload.password, name:  payload.name });
    // Optionally call reload() if you want to force a fetch immediately
    this.signupResource.reload();
  }
}