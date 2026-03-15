@extends('layouts.guest')


@section('content')
<div class="login-wrapper min-vh-100 d-flex align-items-center justify-content-center px-3 pt-5">
    <div class="w-100 position-relative z-index-2 mt-5" style="max-width: 450px;">
        <!-- Card Utama Login -->
        <div class="glass-auth-card animate-up">
            <div class="card-body p-4 p-md-5">
                <!-- Header -->
                <div class="text-center mb-4">
                    <div class="icon-circle bg-rice-green text-white mx-auto mb-3 shadow-sm" style="width: 80px; height: 80px;">
                        <img src="{{ asset('images/logo.png') }}" alt="Logo" class="img-fluid" style="max-width: 60%; height: auto;">
                    </div>
                    <h3 class="fw-bold text-dark mb-1">Welcome Back!</h3>
                    <p class="text-secondary small">Masuk untuk mengelola hasil panen Anda</p>
                </div>

                <!-- Session Status Alert -->
                <x-auth-session-status class="alert alert-success border-0 shadow-sm mb-4 rounded-3 text-center small" :status="session('status')" />

                <form method="POST" action="{{ route('login') }}" class="d-flex flex-column gap-3">
                    @csrf
                    
                    <!-- Email -->
                    <div class="form-group">
                        <label for="email" class="form-label text-dark fw-bold small text-uppercase" style="letter-spacing: 1px;">Email Address</label>
                        <div class="input-group">
                            <span class="input-group-text bg-white border-end-0 text-secondary ps-3 rounded-start-pill"><i class="fas fa-envelope"></i></span>
                            <input id="email" type="email" name="email" value="{{ old('email') }}" required autofocus autocomplete="username" placeholder="name@example.com" class="form-control border-start-0 rounded-end-pill py-3 ps-1 bg-white shadow-sm" style="font-size: 0.95rem;">
                        </div>
                        <x-input-error :messages="$errors->get('email')" class="mt-1 text-danger small" />
                    </div>

                    <!-- Password -->
                    <div class="form-group">
                        <label for="password" class="form-label text-dark fw-bold small text-uppercase" style="letter-spacing: 1px;">Password</label>
                        <div class="input-group">
                            <span class="input-group-text bg-white border-end-0 text-secondary ps-3 rounded-start-pill"><i class="fas fa-lock"></i></span>
                            <input id="password" type="password" name="password" required autocomplete="current-password" placeholder="Enter your password" class="form-control border-start-0 rounded-end-pill py-3 ps-1 bg-white shadow-sm" style="font-size: 0.95rem;">
                        </div>
                        <x-input-error :messages="$errors->get('password')" class="mt-1 text-danger small" />
                    </div>

                    <!-- Remember & Forgot Password -->
                    <div class="d-flex align-items-center justify-content-between mt-2">
                        <div class="form-check">
                            <input id="remember" type="checkbox" name="remember" class="form-check-input border-2 border-secondary" style="cursor: pointer;">
                            <label for="remember" class="form-check-label text-secondary small cursor-pointer fw-semibold user-select-none">Remember me</label>
                        </div>
                        
                        @if (Route::has('password.request'))
                            <a href="{{ route('password.request') }}" class="text-rice-green fw-bold small text-decoration-none hover-underline">
                                Forgot Password?
                            </a>
                        @endif
                    </div>

                    <!-- Submit Button -->
                    <button type="submit" class="btn btn-rice-green w-100 rounded-pill py-3 fw-bold shadow-lg mt-3 hover-scale border-0" style="background: linear-gradient(135deg, #4CAF50, #2E7D32); font-size: 1.1rem;">
                        LOG IN
                    </button>

                    <!-- Link Register -->
                    <div class="text-center mt-4 pt-3 border-top border-secondary-subtle">
                        <p class="text-secondary small mb-2">Belum punya akun?</p>
                        <a href="{{ route('register') }}" class="btn btn-outline-success w-100 rounded-pill py-2 fw-bold small" style="border-width: 2px;">
                            Buat Akun Baru
                        </a>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>

<style>
    /* Page Background */
    .login-wrapper {
        background-image: url('{{ asset('images/Background.png') }}');
        background-size: cover;
        background-position: center bottom;
        background-repeat: no-repeat;
    }

    /* Premium Glass Authentication Card */
    .glass-auth-card {
        background: rgba(255, 255, 255, 0.5); /* 50% Opacity as requested */
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 2px solid rgba(255, 255, 255, 0.6);
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }

    .icon-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #4CAF50, #81C784);
    }
    
    .text-rice-green { color: #2E7D32; }
    .bg-rice-green { background-color: #4CAF50; }

    /* Form Styles */
    .form-control:focus {
        box-shadow: none;
        border-color: #4CAF50;
        background-color: #fff;
    }
    
    .input-group-text {
        border-color: #dee2e6;
    }
    
    .input-group:focus-within .input-group-text {
        border-color: #4CAF50;
        color: #4CAF50 !important;
    }
    
    .input-group:focus-within .form-control {
        border-color: #4CAF50;
    }

    /* Animation */
    .animate-up {
        animation: fadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) both;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .hover-scale { transition: transform 0.2s; }
    .hover-scale:hover { transform: translateY(-2px); }
    
    .hover-underline:hover { text-decoration: underline !important; }
</style>
@endsection