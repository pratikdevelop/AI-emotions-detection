import { CommonModule } from '@angular/common';
import { Component, inject, signal } from '@angular/core';
import {ReactiveFormsModule, FormGroup, FormControl, Validators, FormsModule, ValidatorFn, AbstractControl, ValidationErrors} from '@angular/forms';
import {MatSnackBar, MatSnackBarModule} from '@angular/material/snack-bar'
import axios from 'axios';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon'
import { environment } from '../../environments/environment';
import { AuthService } from '../../services/auth.service';


@Component({
  selector: 'app-signup',
  imports: [ReactiveFormsModule, CommonModule, FormsModule, MatSnackBarModule,  MatFormFieldModule,
    MatInputModule,
    MatIconModule,
    MatButtonModule,],
  templateUrl: './signup.component.html',
  styleUrl: './signup.component.css'
})
export class SignupComponent {
  snackBar = inject(MatSnackBar);
  signupForm: FormGroup = new FormGroup({
    name: new FormControl('avi', [Validators.required, Validators.minLength(3)]),
    email: new FormControl('avi@yopmail.com', [Validators.required, Validators.email]),
    password: new FormControl('Access@#$1234', [Validators.required, Validators.minLength(6)]),
    confirmPassword: new FormControl('Access@#$1234', [
      Validators.required,
      Validators.minLength(6),
      this.passwordMatchValidator(),
    ]),
  });
  passwordVisible =  signal<boolean>(false);
  confirmPasswordVisible = signal<boolean>(false);
  signupError = signal<string>('');
  authService = inject(AuthService);

  // Toggle password visibility
  togglePasswordVisibility(type: string) {
    if (type === 'password') {
      this.passwordVisible.update((prev) => !prev);
    } else {
      this.confirmPasswordVisible.update((prev) => !prev);
    }
      
  }
  
   passwordMatchValidator(): ValidatorFn {
    return (control: AbstractControl): ValidationErrors | null => {
      const password = control.parent?.get('password')?.value;
      const confirmPassword = control.value;
      return password && confirmPassword && password !== confirmPassword
        ? { passwordsMismatch: true }
        : null;
    };
  }

  handleSubmit(): void {
    if (this.signupForm.invalid) return;
    this.authService.signup(this.signupForm.value);
}
    
   
}

