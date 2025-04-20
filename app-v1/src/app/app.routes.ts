import { Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard.component';
import { SignupComponent } from './signup/signup.component';
import { LoginComponent } from './login/login.component';

export const routes: Routes = [
    {
        path: '',
        redirectTo:
            'login',
        pathMatch: 'full'
    },
    {
        path: 'dashboard',
        component: DashboardComponent
    },
    {
        path: 'signup',
        component: SignupComponent
    },
{
    path: 'login',
    component: LoginComponent
}
];
