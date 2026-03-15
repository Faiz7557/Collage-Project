@extends('layouts.guest')

@section('content')
<div class="login-wrapper min-vh-100 d-flex align-items-center justify-content-center px-3 py-5">
    <div class="w-100 position-relative z-index-2 mt-5" style="max-width: 500px;">
        <!-- Card Utama Register -->
        <div class="glass-auth-card animate-up">
            <div class="card-body p-4 p-md-5">
                <!-- Header -->
                <div class="text-center mb-4">
                    <div class="icon-circle bg-rice-green text-white mx-auto mb-3 shadow-sm" style="width: 80px; height: 80px;">
                        <img src="{{ asset('images/logo.png') }}" alt="Logo" class="img-fluid" style="max-width: 60%; height: auto;">
                    </div>
                    <h3 class="fw-bold text-dark mb-1">Create Account</h3>
                    <p class="text-secondary small">Bergabunglah dengan ekosistem Warung Padi</p>
                </div>

                <!-- Session Status Alert -->
                <x-auth-session-status class="alert alert-success border-0 shadow-sm mb-4 rounded-3 text-center small" :status="session('status')" />

                <form method="POST" action="{{ route('register') }}" class="d-flex flex-column gap-3">
                    @csrf
                    
                    <!-- Nama Lengkap -->
                    <div class="form-group">
                        <label for="nama" class="form-label text-dark fw-bold small text-uppercase" style="letter-spacing: 1px;">Full Name</label>
                        <div class="input-group">
                            <span class="input-group-text bg-white border-end-0 text-secondary ps-3 rounded-start-pill"><i class="fas fa-user"></i></span>
                            <input id="nama" type="text" name="nama" value="{{ old('nama') }}" required autofocus autocomplete="nama" placeholder="Your Full Name" class="form-control border-start-0 rounded-end-pill py-3 ps-1 bg-white shadow-sm" style="font-size: 0.95rem;">
                        </div>
                        <x-input-error :messages="$errors->get('nama')" class="mt-1 text-danger small" />
                    </div>

                    <!-- Email -->
                    <div class="form-group">
                        <label for="email" class="form-label text-dark fw-bold small text-uppercase" style="letter-spacing: 1px;">Email Address</label>
                        <div class="input-group">
                            <span class="input-group-text bg-white border-end-0 text-secondary ps-3 rounded-start-pill"><i class="fas fa-envelope"></i></span>
                            <input id="email" type="email" name="email" value="{{ old('email') }}" required autocomplete="username" placeholder="name@example.com" class="form-control border-start-0 rounded-end-pill py-3 ps-1 bg-white shadow-sm" style="font-size: 0.95rem;">
                        </div>
                        <x-input-error :messages="$errors->get('email')" class="mt-1 text-danger small" />
                    </div>

                    <!-- Peran -->
                    <div class="form-group">
                        <label for="peran" class="form-label text-dark fw-bold small text-uppercase" style="letter-spacing: 1px;">Register As</label>
                        <div class="input-group">
                            <span class="input-group-text bg-white border-end-0 text-secondary ps-3 rounded-start-pill"><i class="fas fa-users"></i></span>
                            <select id="peran" name="peran" required class="form-select border-start-0 rounded-end-pill py-3 ps-1 bg-white shadow-sm cursor-pointer" style="font-size: 0.95rem;">
                                <option value="" disabled selected>Select Role</option>
                                <option value="petani" {{ old('peran') == 'petani' ? 'selected' : '' }}>Petani</option>
                                <option value="pengepul" {{ old('peran') == 'pengepul' ? 'selected' : '' }}>Pengepul</option>
                                <option value="distributor" {{ old('peran') == 'distributor' ? 'selected' : '' }}>Distributor</option>
                            </select>
                        </div>
                        <x-input-error :messages="$errors->get('peran')" class="mt-1 text-danger small" />
                    </div>

                    <!-- Password -->
                    <div class="form-group">
                        <label for="password" class="form-label text-dark fw-bold small text-uppercase" style="letter-spacing: 1px;">Password</label>
                        <div class="input-group">
                            <span class="input-group-text bg-white border-end-0 text-secondary ps-3 rounded-start-pill"><i class="fas fa-lock"></i></span>
                            <input id="password" type="password" name="password" required autocomplete="new-password" placeholder="Create a strong password" class="form-control border-start-0 rounded-end-pill py-3 ps-1 bg-white shadow-sm" style="font-size: 0.95rem;">
                        </div>
                        <x-input-error :messages="$errors->get('password')" class="mt-1 text-danger small" />
                    </div>

                    <!-- Konfirmasi Password -->
                    <div class="form-group">
                        <label for="password_confirmation" class="form-label text-dark fw-bold small text-uppercase" style="letter-spacing: 1px;">Confirm Password</label>
                        <div class="input-group">
                            <span class="input-group-text bg-white border-end-0 text-secondary ps-3 rounded-start-pill"><i class="fas fa-check-circle"></i></span>
                            <input id="password_confirmation" type="password" name="password_confirmation" required autocomplete="new-password" placeholder="Repeat your password" class="form-control border-start-0 rounded-end-pill py-3 ps-1 bg-white shadow-sm" style="font-size: 0.95rem;">
                        </div>
                        <x-input-error :messages="$errors->get('password_confirmation')" class="mt-1 text-danger small" />
                    </div>

                    <!-- Submit Button -->
                    <button type="submit" class="btn btn-rice-green w-100 rounded-pill py-3 fw-bold shadow-lg mt-3 hover-scale border-0" style="background: linear-gradient(135deg, #4CAF50, #2E7D32); font-size: 1.1rem;">
                        SIGN UP
                    </button>

                    <!-- Link Login -->
                    <div class="text-center mt-4 pt-3 border-top border-secondary-subtle">
                        <p class="text-secondary small mb-2">Sudah punya akun?</p>
                        <a href="{{ route('login') }}" class="btn btn-outline-success w-100 rounded-pill py-2 fw-bold small" style="border-width: 2px;">
                            Masuk Sekarang
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
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 2px solid rgba(255, 255, 255, 0.6);
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }

    .icon-circle {
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #4CAF50, #81C784);
    }
    
    .text-rice-green { color: #2E7D32; }
    .bg-rice-green { background-color: #4CAF50; }

    /* Form Styles */
    .form-control:focus, .form-select:focus {
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
    
    .input-group:focus-within .form-control, .input-group:focus-within .form-select {
        border-color: #4CAF50;
    }

    .form-select {
        background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill='none' stroke='%23343a40' stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='m2 5 6 6 6-6'/%3e%3c/svg%3e");
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
</style>
@endsection